import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { isRailShortcut, readRailState, type ShortcutKey } from "./rail.ts";

function storage(value: string | null) {
  return { getItem: () => value };
}

describe("readRailState", () => {
  it("restores an expanded rail", () => {
    assert.equal(readRailState(storage("expanded")), "expanded");
  });

  it("defaults to collapsed when nothing or something unknown is stored", () => {
    assert.equal(readRailState(storage(null)), "collapsed");
    assert.equal(readRailState(storage("wide")), "collapsed");
    assert.equal(readRailState(undefined), "collapsed");
  });

  it("defaults to collapsed when storage throws", () => {
    const blocked = {
      getItem: () => {
        throw new DOMException("denied", "SecurityError");
      },
    };
    assert.equal(readRailState(blocked), "collapsed");
  });
});

describe("isRailShortcut", () => {
  const bracket: ShortcutKey = { key: "[", metaKey: false, ctrlKey: false, altKey: false, repeat: false, defaultPrevented: false };

  it("fires on a bare [", () => {
    assert.equal(isRailShortcut(bracket, false), true);
  });

  it("ignores [ while typing", () => {
    assert.equal(isRailShortcut(bracket, true), false);
  });

  it("ignores modified, repeated, handled, and other keys", () => {
    assert.equal(isRailShortcut({ ...bracket, metaKey: true }, false), false);
    assert.equal(isRailShortcut({ ...bracket, ctrlKey: true }, false), false);
    assert.equal(isRailShortcut({ ...bracket, altKey: true }, false), false);
    assert.equal(isRailShortcut({ ...bracket, repeat: true }, false), false);
    assert.equal(isRailShortcut({ ...bracket, defaultPrevented: true }, false), false);
    assert.equal(isRailShortcut({ ...bracket, key: "]" }, false), false);
  });
});
