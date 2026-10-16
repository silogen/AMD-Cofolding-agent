// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error("index.html is missing the #root element");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
