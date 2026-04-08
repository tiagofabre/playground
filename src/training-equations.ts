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

/** Scalar for TeX \text{...} (matches network-equation numeric style). */
export function fmtCoeff(n: number): string {
  let s = n.toFixed(3);
  if (s === "-0.000") {
    return "0.000";
  }
  return s;
}

/**
 * Total objective: mean half-square error plus optional L1/L2 on weights
 * (see nn.updateWeights and RegularizationFunction in nn.ts).
 */
export function buildObjectiveTex(
    regularizationKey: string | undefined,
    lambda: number): string {
  let key = regularizationKey === "L1" || regularizationKey === "L2" ?
      regularizationKey :
      "none";
  let dataMean =
      "\\frac{1}{N}\\sum_{i=1}^{N} \\frac{1}{2}\\left(\\hat{y}_i - " +
      "y_i\\right)^2";
  if (key === "none") {
    return "\\begin{aligned}\n\\mathcal{L} &= " + dataMean + "\n\\end{aligned}";
  }
  let regTerm = key === "L1" ?
      "\\lambda \\sum_{w} \\lvert w\\rvert" :
      "\\lambda \\sum_{w} \\frac{1}{2} w^2";
  let lam = fmtCoeff(lambda);
  return (
      "\\begin{aligned}\n\\mathcal{L} &= " + dataMean + " + " + regTerm +
      "\\\\\n&\\quad \\lambda = \\text{" + lam +
      "} \\text{ (regularization rate)}\n\\end{aligned}");
}

/** Numeric values for the output neuron and one incoming weight (see extract). */
export interface BackpropNumericSnapshot {
  yHat: number;
  y: number;
  e: number;
  dEdYhat: number;
  zOut: number;
  sigmaPrimeOut: number;
  deltaOut: number;
  exampleWeight: number;
  exampleAFrom: number;
  exampleDEdW: number;
  hiddenDelta?: number;
  hiddenZ?: number;
  hiddenSigmaPrime?: number;
  hiddenDEdA?: number;
}

/**
 * Read derivatives and activations right after nn.backProp for this example.
 * Caller must ensure forwardProp+backProp just ran for the same point (before
 * any further forwardProp, e.g. from getLoss).
 */
export function extractBackpropSnapshot(
    network: nn.Node[][], labelY: number): BackpropNumericSnapshot | null {
  if (network == null || network.length < 2) {
    return null;
  }
  let out = network[network.length - 1][0];
  if (out == null) {
    return null;
  }
  let yHat = out.output;
  let y = labelY;
  let e = 0.5 * (yHat - y) * (yHat - y);
  let zOut = out.totalInput;
  let sigmaPrimeOut = out.activation.der(zOut);
  let dEdYhat = out.outputDer;
  let deltaOut = out.inputDer;

  let linkToShow: nn.Link | null = null;
  for (let j = 0; j < out.inputLinks.length; j++) {
    let L = out.inputLinks[j];
    if (!L.isDead) {
      linkToShow = L;
      break;
    }
  }
  if (linkToShow == null && out.inputLinks.length > 0) {
    linkToShow = out.inputLinks[0];
  }
  let exampleWeight = linkToShow != null ? linkToShow.weight : 0;
  let exampleAFrom = linkToShow != null && linkToShow.source != null ?
      linkToShow.source.output :
      0;
  let exampleDEdW = linkToShow != null ? linkToShow.errorDer : 0;

  let snap: BackpropNumericSnapshot = {
    yHat: yHat,
    y: y,
    e: e,
    dEdYhat: dEdYhat,
    zOut: zOut,
    sigmaPrimeOut: sigmaPrimeOut,
    deltaOut: deltaOut,
    exampleWeight: exampleWeight,
    exampleAFrom: exampleAFrom,
    exampleDEdW: exampleDEdW
  };

  if (network.length > 2) {
    let h0 = network[1][0];
    if (h0 != null) {
      snap.hiddenDelta = h0.inputDer;
      snap.hiddenZ = h0.totalInput;
      snap.hiddenSigmaPrime = h0.activation.der(h0.totalInput);
      snap.hiddenDEdA = h0.outputDer;
    }
  }
  return snap;
}

/**
 * Backprop for scalar output and per-example squared error (nn.backProp with
 * Errors.SQUARE). Optional snapshot from the last training example in an epoch.
 */
export function buildBackpropTex(snapshot: BackpropNumericSnapshot | null): string {
  let f = fmtCoeff;
  let rows: string[] = [
    "E &= \\tfrac{1}{2}\\left(\\hat{y} - y\\right)^2 " +
        "\\quad \\text{(one example)}\\\\",
    "\\frac{\\partial E}{\\partial \\hat{y}} &= \\hat{y} - y\\\\",
    "\\delta^{(z)} &= \\frac{\\partial E}{\\partial z} = " +
        "\\frac{\\partial E}{\\partial a}\\,\\sigma'(z), \\quad " +
        "a = \\sigma(z)\\\\",
    "\\frac{\\partial E}{\\partial w} &= \\delta^{(z)}_{\\text{to}}\\, " +
        "a_{\\text{from}}\\\\",
    "\\frac{\\partial E}{\\partial a_{\\text{from}}} &= " +
        "\\sum_{\\text{out}} w\\,\\delta^{(z)}_{\\text{out}}"
  ];
  if (snapshot != null) {
    let s = snapshot;
    rows.push(
        "&\\text{Values after the last training example in the latest step:}\\\\",
        "E &= \\tfrac{1}{2}(\\hat{y}-y)^2 = \\text{" + f(s.e) + "} \\quad " +
            "(\\hat{y}=\\text{" + f(s.yHat) + "},\\, y=\\text{" + f(s.y) + "})\\\\",
        "\\frac{\\partial E}{\\partial \\hat{y}} &= \\text{" + f(s.dEdYhat) +
            "} \\quad (z_{\\text{out}}=\\text{" + f(s.zOut) +
            "},\\, \\sigma'(z_{\\text{out}})=\\text{" + f(s.sigmaPrimeOut) + "})\\\\",
        "\\delta^{(z)}_{\\text{out}} &= \\text{" + f(s.deltaOut) + "}\\\\",
        "\\frac{\\partial E}{\\partial w} &= \\text{" + f(s.exampleDEdW) +
            "} \\quad \\text{(first non-dead input to output; } w=\\text{" +
            f(s.exampleWeight) + "},\\, a_{\\text{from}}=\\text{" +
            f(s.exampleAFrom) + "})");
    if (s.hiddenDelta != null && s.hiddenZ != null &&
        s.hiddenSigmaPrime != null && s.hiddenDEdA != null) {
      rows.push(
          "\\delta^{(z)}_{h_1} &= \\text{" + f(s.hiddenDelta) + "} \\quad " +
              "(z=\\text{" + f(s.hiddenZ) + "},\\, \\sigma'(z)=\\text{" +
              f(s.hiddenSigmaPrime) + "})\\\\",
          "\\frac{\\partial E}{\\partial a_{h_1}} &= \\text{" +
              f(s.hiddenDEdA) + "}");
    }
  } else {
    rows.push(
        "&\\text{Train with Step or Play to show numbers from the last example " +
            "in each epoch.}");
  }
  return "\\begin{aligned}\n" + rows.join("\\\\\n") + "\n\\end{aligned}";
}

/**
 * nn.updateWeights: averaged gradient step, then optional regularization step.
 */
export function buildWeightUpdateTex(
    learningRate: number,
    lambda: number,
    regularizationKey: string | undefined): string {
  let eta = fmtCoeff(learningRate);
  let key = regularizationKey === "L1" || regularizationKey === "L2" ?
      regularizationKey :
      "none";
  let rows: string[] = [
    "w &\\leftarrow w - \\frac{\\eta}{n_{\\text{b}}}\\,\\overline{" +
        "\\frac{\\partial E}{\\partial w}}\\\\",
    "b &\\leftarrow b - \\frac{\\eta}{n_{\\text{b}}}\\,\\overline{" +
        "\\frac{\\partial E}{\\partial b}}\\\\",
    "&\\text{where } \\overline{\\cdot} \\text{ is the mean over the " +
        "mini-batch of size } n_{\\text{b}} \\text{; } \\eta = \\text{" +
        eta + "} \\text{ (learning rate).}"
  ];
  if (key !== "none") {
    let lam = fmtCoeff(lambda);
    rows.push(
        "w &\\leftarrow w - \\eta\\lambda\\, R'(w) \\quad \\text{(after " +
            "the step above; } R' \\text{ uses } w \\text{ from that step)}");
    if (key === "L1") {
      rows.push(
          "&\\lambda = \\text{" + lam +
          "},\\ R(w)=\\lvert w\\rvert,\\ R'(w)\\in\\{-1,0,1\\} " +
          "\\text{(0 at }w=0\\text{). L1 may set } w\\to 0 \\text{ when " +
          "the update would change sign.}");
    } else {
      rows.push(
          "&\\lambda = \\text{" + lam +
          "},\\ R(w)=\\tfrac{1}{2}w^2,\\ R'(w)=w.");
    }
  }
  return "\\begin{aligned}\n" + rows.join("\\\\\n") + "\n\\end{aligned}";
}
