import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { splitTokenFragment, takeTokenFromUrl } from "./tokenFragment.ts";

function fakeHistory() {
  const urls: string[] = [];
  return { urls, state: { idx: 3 }, replaceState: (_d: unknown, _u: string, url: string) => urls.push(url) };
}

describe("splitTokenFragment", () => {
  it("reads the token and keeps any other fragment", () => {
    assert.deepEqual(splitTokenFragment("#token=abc-_123"), { token: "abc-_123", rest: "" });
    assert.deepEqual(splitTokenFragment("#token=abc&tab=runs"), { token: "abc", rest: "#tab=runs" });
  });

  it("leaves a fragment without a token alone", () => {
    assert.deepEqual(splitTokenFragment(""), { token: null, rest: "" });
    assert.deepEqual(splitTokenFragment("#kernels"), { token: null, rest: "#kernels" });
    assert.deepEqual(splitTokenFragment("#token="), { token: null, rest: "#token=" });
  });
});

describe("takeTokenFromUrl", () => {
  it("returns the token and strips it from the address bar", () => {
    const history = fakeHistory();
    const token = takeTokenFromUrl({ pathname: "/runs", search: "?x=1", hash: "#token=s3cret" }, history);
    assert.equal(token, "s3cret");
    assert.deepEqual(history.urls, ["/runs?x=1"]);
  });

  it("does not touch history when there is no token", () => {
    const history = fakeHistory();
    assert.equal(takeTokenFromUrl({ pathname: "/", search: "", hash: "#a" }, history), null);
    assert.deepEqual(history.urls, []);
  });
});
