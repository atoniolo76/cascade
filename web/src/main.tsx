import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router";
import "./index.css";
import "./lib/theme";
import { ApiError, tokenStore } from "./api/client";
import { AppShell } from "./components/AppShell";
import { TokenGate } from "./components/TokenGate";
import { TracesPage } from "./pages/TracesPage";
import { TraceDetailPage } from "./pages/TraceDetailPage";
import { RunsPage } from "./pages/RunsPage";
import { NewRunPage } from "./pages/NewRunPage";
import { NotFoundPage } from "./pages/NotFoundPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 2,
    },
  },
});

tokenStore.subscribe(() => {
  if (tokenStore.get() === null) queryClient.clear();
});

const router = createBrowserRouter([
  {
    element: <AppShell />,
    hydrateFallbackElement: <div className="min-h-dvh bg-window" />,
    children: [
      { index: true, element: <TracesPage /> },
      { path: "traces/:traceId", element: <TraceDetailPage /> },
      { path: "runs", element: <RunsPage /> },
      { path: "runs/new", element: <NewRunPage /> },
      { path: "request-simulator", lazy: () => import("./pages/RequestSimulatorPage").then((m) => ({ Component: m.RequestSimulatorPage })) },
      { path: "runs/:runId", lazy: () => import("./pages/RunDetailPage").then((m) => ({ Component: m.RunDetailPage })) },
      { path: "compare", lazy: () => import("./pages/ComparePage").then((m) => ({ Component: m.ComparePage })) },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);

const root = document.getElementById("root");
if (root === null) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TokenGate>
        <RouterProvider router={router} />
      </TokenGate>
    </QueryClientProvider>
  </StrictMode>,
);
