/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";

/** Formats a scalar for LaTeX (fixed decimals, trim negative zero). */
function fmt(n: number): string {
  let s = n.toFixed(3);
  if (s === "-0.000") {
    return "0.000";
  }
  return s;
}

/** Builds bias + sum_k w_k * symbol_k with clean + / - spacing. */
function affineTex(node: nn.Node, prevTex: string[]): string {
  let parts: string[] = [fmt(node.bias)];
  for (let j = 0; j < node.inputLinks.length; j++) {
    parts.push(fmt(node.inputLinks[j].weight) + " \\cdot " + prevTex[j]);
  }
  return parts.join(" + ").replace(/\+ -/g, "- ");
}

/** Maps activation keys (as in state.activations) to LaTeX. */
function actWrap(activationKey: string, body: string): string {
  switch (activationKey) {
    case "relu":
      return "\\operatorname{ReLU}\\left(" + body + "\\right)";
    case "tanh":
      return "\\tanh\\left(" + body + "\\right)";
    case "sigmoid":
      return "\\sigma\\left(" + body + "\\right)";
    case "linear":
      return body;
    default:
      return "\\left(" + body + "\\right)";
  }
}

function sigmaLegendRow(): string {
  return "&\\text{with } \\sigma(t)=\\dfrac{1}{1+e^{-t}}";
}

function needsSigmaLegend(hiddenKey: string, outputKey: string): boolean {
  return hiddenKey === "sigmoid" || outputKey === "sigmoid";
}

/**
 * Layer-wise definition: equations reference h_j^{(l)} symbols.
 */
function buildLayeredTex(
    network: nn.Node[][], inputSymbols: string[],
    hiddenActivationKey: string, outputActivationKey: string): string {
  let lines: string[] = [];
  let prevTex = inputSymbols.slice();

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let layer = network[layerIdx];
    let isOutput = layerIdx === network.length - 1;
    let actKey = isOutput ? outputActivationKey : hiddenActivationKey;
    let nextPrev: string[] = [];

    for (let i = 0; i < layer.length; i++) {
      let node = layer[i];
      let inner = affineTex(node, prevTex);
      let rhs = actWrap(actKey, inner);

      if (isOutput) {
        lines.push("\\hat{y} &= " + rhs);
      } else {
        let h = "h_{" + (i + 1) + "}^{(" + layerIdx + ")}";
        nextPrev.push(h);
        lines.push(h + " &= " + rhs);
      }
    }
    if (!isOutput) {
      prevTex = nextPrev;
    }
  }

  if (needsSigmaLegend(hiddenActivationKey, outputActivationKey)) {
    lines.push(sigmaLegendRow());
  }
  return "\\begin{aligned}\n" + lines.join("\\\\\n") + "\n\\end{aligned}";
}

/**
 * Layer-wise MLP formula with numeric weights and biases.
 * inputSymbols must match network[0] order (same as constructInput / buildNetwork).
 */
export function buildNetworkEquationTex(
    network: nn.Node[][], inputSymbols: string[],
    hiddenActivationKey: string, outputActivationKey: string): string {
  if (network == null || network.length < 2) {
    return "";
  }
  if (inputSymbols.length !== network[0].length) {
    return "";
  }
  return buildLayeredTex(
      network, inputSymbols, hiddenActivationKey, outputActivationKey);
}
