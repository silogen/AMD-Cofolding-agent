// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { ComponentProps } from "react";
import { Editor } from "ketcher-react";
import { StandaloneStructServiceProvider } from "ketcher-standalone";
import type { Ketcher, StructServiceProvider } from "ketcher-core";
import "ketcher-react/dist/index.css";
import "./sketcher.css";

// One chemistry engine (Indigo in WebAssembly) for every opening of the sketcher.
const structServiceProvider = new StandaloneStructServiceProvider() as unknown as StructServiceProvider;

type Buttons = NonNullable<ComponentProps<typeof Editor>["buttons"]>;

/** Buttons that lead away from drawing one small molecule. */
const HIDDEN: (keyof Buttons)[] = [
  "help",
  "about",
  "miew",
  "settings",
  "fullscreen",
  "sgroup",
  "rgroup",
  "reaction-plus",
  "arrows",
  "reaction-mapping-tools",
  "shape",
  "text",
  "create-monomer",
];
const BUTTONS: Buttons = Object.fromEntries(HIDDEN.map((name) => [name, { hidden: true }]));

interface Props {
  initial?: string;
  onReady: (ketcher: Ketcher) => void;
  onError: (message: string) => void;
}

export default function KetcherEditor({ initial, onReady, onError }: Props) {
  return (
    <Editor
      staticResourcesUrl=""
      structServiceProvider={structServiceProvider}
      errorHandler={onError}
      disableMacromoleculesEditor
      buttons={BUTTONS}
      onInit={(ketcher: Ketcher) => {
        const start = initial ? ketcher.setMolecule(initial) : Promise.resolve();
        start.then(
          () => onReady(ketcher),
          () => {
            onError(`The sketcher could not read ${initial}.`);
            onReady(ketcher);
          },
        );
      }}
    />
  );
}
