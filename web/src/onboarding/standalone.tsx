import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../index.css";
import "../lib/theme";
import { OnboardingPage } from "./OnboardingPage";

/** The public demo on cascadesim.vercel.app/onboarding: no workspace yet, so it ends at Get started. */
const root = document.getElementById("root");
if (root === null) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <OnboardingPage
      finishLabel="Create your workspace"
      onFinish={() => {
        window.location.assign("/start.html");
      }}
    />
  </StrictMode>,
);
