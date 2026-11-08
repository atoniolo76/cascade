import { Link } from "react-router";
import { Card, EmptyState } from "../components/ui";
import { paths } from "../paths";

export function NotFoundPage() {
  return (
    <Card>
      <EmptyState
        title="Nothing here"
        body="This page doesn't exist."
        action={
          <Link to={paths.traces} className="text-[13px] font-medium text-accent-ink hover:underline">
            Back to traces
          </Link>
        }
      />
    </Card>
  );
}
