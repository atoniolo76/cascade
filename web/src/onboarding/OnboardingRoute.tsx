import { useNavigate } from "react-router";
import { OnboardingPage } from "./OnboardingPage";

/** Inside a workspace, finishing onboarding opens the app. */
export function OnboardingRoute() {
  const navigate = useNavigate();
  return (
    <OnboardingPage
      onFinish={() => {
        void navigate("/");
      }}
    />
  );
}
