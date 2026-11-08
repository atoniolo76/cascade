import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { chooseEndpoint } from "./endpoint.ts";

describe("chooseEndpoint", () => {
  it("prefills the endpoint the app reports", () => {
    assert.equal(chooseEndpoint({ typed: null, reported: "https://sb-1-9000.sail.box", fallback: "" }), "https://sb-1-9000.sail.box");
  });

  it("keeps what the user typed over the reported endpoint", () => {
    assert.equal(chooseEndpoint({ typed: "https://s3.mine.dev", reported: "https://sb-1-9000.sail.box", fallback: "" }), "https://s3.mine.dev");
  });

  it("keeps an endpoint the user cleared on purpose", () => {
    assert.equal(chooseEndpoint({ typed: "", reported: "https://sb-1-9000.sail.box", fallback: "" }), "");
  });

  it("falls back when the app doesn't know its endpoint", () => {
    assert.equal(chooseEndpoint({ typed: null, reported: null, fallback: "http://localhost:9000" }), "http://localhost:9000");
  });
});
