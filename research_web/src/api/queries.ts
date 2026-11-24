// Every server call the UI makes, as TanStack Query hooks. The query cache makes revisiting a page
// instant; job lists poll ONLY while something is running.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, post, postStream } from "./client";
import type { AskResult, ChatInfo, JobDetails, JobInfo, Readiness, ResearchReport, SearchHit, TopicSummary, UsageInfo,
  UserInfo } from "./types";

export const keys = {
  me: ["me"] as const,
  topics: ["topics"] as const,
  topic: (id: number) => ["topic", id] as const,
  jobs: ["jobs"] as const,
};

export const isActive = (job: JobInfo) => job.status === "queued" || job.status === "running";

export function useMe() {
  return useQuery({ queryKey: keys.me, queryFn: () => api<UserInfo>("/api/auth/me"), retry: false });
}

export function useTopics() {
  return useQuery({ queryKey: keys.topics, queryFn: () => api<TopicSummary[]>("/api/topics") });
}

export function useJobs() {
  return useQuery({
    queryKey: keys.jobs,
    queryFn: () => api<JobInfo[]>("/api/jobs"),
    refetchInterval: (query) => (query.state.data?.some(isActive) ? 2000 : false),
  });
}

/** When ANY run finishes (wherever you are in the app): refresh the topic list, your usage, and the topic
 *  it made or analysed, so new results simply appear. Used once, in the layout. */
export function useRefreshWhenJobsFinish() {
  const jobs = useJobs();
  const client = useQueryClient();
  const active = useRef(new Set<number>());
  useEffect(() => {
    if (!jobs.data) return;
    const now = new Set(jobs.data.filter(isActive).map((j) => j.id));
    for (const id of active.current) {
      if (now.has(id)) continue;
      const job = jobs.data.find((j) => j.id === id);
      client.invalidateQueries({ queryKey: keys.topics });
      client.invalidateQueries({ queryKey: ["usage"] });
      client.invalidateQueries({ queryKey: ["job", id] });
      client.invalidateQueries({ queryKey: ["readiness"] });         // an "index" job prepares a topic
      if (job?.topic_id) client.invalidateQueries({ queryKey: keys.topic(job.topic_id) });
    }
    active.current = now;
  }, [jobs.data, client]);
}

/** One job, followed LIVE while it is queued or running (SSE step 6): the server pushes every change down
 *  /api/jobs/{id}/events. If that stream can't connect, it falls back to asking every 1.5 s as before. */
export function useJob(id: number | null) {
  const client = useQueryClient();
  const enabled = id !== null && Number.isFinite(id);
  const [streaming, setStreaming] = useState(false);
  const query = useQuery({
    queryKey: ["job", id],
    queryFn: () => api<JobInfo>(`/api/jobs/${id}`),
    enabled,
    refetchInterval: (q) => (streaming || (q.state.data && !isActive(q.state.data)) ? false : 1500),
  });
  const finished = query.data ? !isActive(query.data) : false;
  useEffect(() => {
    if (!enabled || finished || typeof EventSource === "undefined") return;
    // a GET with the login cookie: the browser's own EventSource works, and reconnects by itself when the
    // server ends a stream (it does every minute, see jobs_controller.py)
    const source = new EventSource(`/api/jobs/${id}/events`);
    source.onopen = () => setStreaming(true);
    source.onerror = () => { if (source.readyState === EventSource.CLOSED) setStreaming(false); };  // gave up: poll
    source.addEventListener("job", (event) => {
      const job = JSON.parse((event as MessageEvent).data) as JobInfo;
      client.setQueryData(["job", id], job);
      if (!isActive(job)) {
        source.close();                                       // done or failed: nothing more will come
        setStreaming(false);
        client.invalidateQueries({ queryKey: keys.jobs });    // the job list (and what it refreshes) at once
      }
    });
    return () => { source.close(); setStreaming(false); };
  }, [id, enabled, finished, client]);
  return query;
}

export function useUsage() {
  // every minute too: a provider that starts failing (free tier used up) shows while a job still runs
  return useQuery({ queryKey: ["usage"], queryFn: () => api<UsageInfo>("/api/usage"), refetchInterval: 60_000 });
}

/** How many papers a research picks, reads in full text (where free) and prepares for chat. */
export type PaperCount = 5 | 10 | 15 | 20;

export function useStartResearch() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ topic, papers }: { topic: string; papers: PaperCount }) =>
      post<JobInfo>("/api/research", { topic, papers }),
    onSuccess: (job) => {
      client.setQueryData(["job", job.id], job);
      client.invalidateQueries({ queryKey: keys.jobs });
    },
  });
}

export function useRetryJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => post<JobInfo>(`/api/jobs/${id}/retry`),
    onSuccess: (job) => {
      client.setQueryData(["job", job.id], job);
      client.invalidateQueries({ queryKey: keys.jobs });
    },
  });
}

export const fetchTopic = (id: number) => api<ResearchReport>(`/api/topics/${id}`);

export function useTopic(id: number) {
  return useQuery({ queryKey: keys.topic(id), queryFn: () => fetchTopic(id), enabled: Number.isFinite(id) });
}

/** Start an analysis (claims, quality, connections, gaps) of a topic as a background job. */
export function useAnalyze(topicId: number) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (refresh: boolean) => post<JobInfo>(`/api/topics/${topicId}/analyze`, { refresh }),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.jobs }),
  });
}

/** Load a topic while the mouse is still on its link: the click then opens it instantly. */
export function usePrefetchTopic() {
  const client = useQueryClient();
  return (id: number) =>
    client.prefetchQuery({ queryKey: keys.topic(id), queryFn: () => fetchTopic(id), staleTime: 60_000 });
}

export function useLogin() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: { email: string; password: string }) => post<UserInfo>("/api/auth/login", body),
    onSuccess: (user) => {
      client.clear(); // nothing from a previous user may remain in the cache
      client.setQueryData(keys.me, user);
    },
  });
}

export function useRegister() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: { email: string; password: string; display_name?: string }) =>
      post<UserInfo>("/api/auth/register", body),
    onSuccess: (user) => {
      client.clear();
      client.setQueryData(keys.me, user);
    },
  });
}

export function useLogout() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => post<void>("/api/auth/logout"),
    onSettled: () => client.clear(),
  });
}

// ---------- questions, chats, search (step 7.6) ----------

export function useReadiness(topicId: number) {
  return useQuery({ queryKey: ["readiness", topicId], queryFn: () => api<Readiness>(`/api/topics/${topicId}/readiness`) });
}

export function usePrepare(topicId: number) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => post<JobInfo>(`/api/topics/${topicId}/prepare`),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.jobs }),
  });
}

export function useChats(topicId?: number) {
  return useQuery({
    queryKey: ["chats", topicId ?? "all"],
    queryFn: () => api<ChatInfo[]>(topicId ? `/api/chats?topic_id=${topicId}` : "/api/chats"),
  });
}

export function useChat(chatId: number | null) {
  return useQuery({
    queryKey: ["chat", chatId],
    queryFn: () => api<ChatInfo>(`/api/chats/${chatId}`),
    enabled: chatId !== null && Number.isFinite(chatId),
  });
}

export function useAsk(topicId: number) {
  const client = useQueryClient();
  return useMutation({
    // streamed: onEvent hears each event as it arrives (stage, text, sentence, reset); resolves with `done`
    mutationFn: ({ onEvent, ...body }: { question: string; chat_id?: number | null;
                                         onEvent?: (event: string, data: unknown) => void }) =>
      postStream<AskResult>(`/api/topics/${topicId}/ask/stream`, body, onEvent),
    onSuccess: (result) => {
      // append to the cached chat at once (no refetch needed), then refresh the lists
      client.setQueryData<ChatInfo>(["chat", result.chat_id], (old) =>
        old ? { ...old, messages: [...(old.messages ?? []), result.question, result.answer] } : old);
      client.invalidateQueries({ queryKey: ["chat", result.chat_id] });
      client.invalidateQueries({ queryKey: ["chats"] });
      client.invalidateQueries({ queryKey: ["usage"] });
    },
  });
}

export function useDeleteChat() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (chatId: number) => api<void>(`/api/chats/${chatId}`, { method: "DELETE" }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["chats"] }),
  });
}

export function useSearch(query: string) {
  return useQuery({
    queryKey: ["search", query],
    queryFn: () => api<SearchHit[]>(`/api/search?q=${encodeURIComponent(query)}`),
    enabled: query.trim().length >= 2,
    staleTime: 5 * 60_000,
  });
}

/** What happened inside a job (LLM use per step + traced prompts). Loaded only when `enabled` (panel open);
 *  refreshed every few seconds while the job still runs. */
export function useJobDetails(id: number, enabled: boolean, running: boolean) {
  return useQuery({
    queryKey: ["job", id, "details"],
    queryFn: () => api<JobDetails>(`/api/jobs/${id}/details`),
    enabled,
    refetchInterval: running ? 4000 : false,
  });
}
