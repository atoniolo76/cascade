import { Link } from "react-router";
import { Card, EmptyState } from "../components/ui";
import { buttonClass } from "../components/styles";
import { paths } from "../paths";

export function NotFoundPage() {
  return (
    <Card>
      <EmptyState
        title="Nothing here"
        body="This page doesn't exist."
        action={
          <Link to={paths.traces} className={buttonClass({})}>
            Back to traces
          </Link>
        }
      />
    </Card>
  );
}
