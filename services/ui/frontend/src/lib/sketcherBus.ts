// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Lets any panel open the molecule sketcher: "Edit in sketcher" on a
 * molecule card, a SMILES chip in the composer, the composer's Sketch button.
 * The composer hosts the sketcher, since what the user draws goes into the
 * message.
 */
export interface SketchRequest {
  /** Start from this molecule; an empty canvas when unset. */
  smiles?: string;
  /** Replace this SMILES of the message with the result, instead of inserting it. */
  replace?: string;
}

type Listener = (request: SketchRequest) => void;

const listeners = new Set<Listener>();

export function openSketcher(smiles?: string, replace?: string): void {
  listeners.forEach((listener) => listener({ smiles, replace }));
}

export function onSketchRequest(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
