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

/** e.g. 6 -> "6th", 21 -> "21st" (for hidden-layer column wording). */
function ordinalSuffix(n: number): string {
  let mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 13) {
    return n + "th";
  }
  switch (n % 10) {
    case 1:
      return n + "st";
    case 2:
      return n + "nd";
    case 3:
      return n + "rd";
    default:
      return n + "th";
  }
}

export interface LegendRow {
  /** Small KaTeX fragment; omit for note-only rows. */
  symbolTex?: string;
  detail: string;
}

export interface LegendSection {
  title: string;
  rows: LegendRow[];
}

const FEATURE_GUIDE: {[id: string]: string} = {
  "x": "Horizontal coordinate of each data point (matches a node in the Features column, top to bottom).",
  "y": "Vertical coordinate of each data point.",
  "xSquared": "Square of the horizontal coordinate.",
  "ySquared": "Square of the vertical coordinate.",
  "xTimesY": "Product of horizontal and vertical coordinates.",
  "sinX": "Sine of the horizontal coordinate.",
  "sinY": "Sine of the vertical coordinate.",
  "cosX": "Cosine of the horizontal coordinate.",
  "cosY": "Cosine of the vertical coordinate."
};

/**
 * Human-readable mapping between equation symbols and the diagram (columns and
 * neuron order top-to-bottom) for the layered equation view.
 */
export function buildEquationLegendSections(
    network: nn.Node[][], inputIds: string[], inputSymbols: string[],
    hiddenActivationSummary: string,
    outputActivationSummary: string): LegendSection[] {
  let sections: LegendSection[] = [];

  if (network == null || network.length < 2) {
    return sections;
  }

  let inputRows: LegendRow[] = [];
  for (let i = 0; i < inputIds.length; i++) {
    let id = inputIds[i];
    let guide = FEATURE_GUIDE[id] != null ?
        FEATURE_GUIDE[id] :
        "Enabled input feature in the Features column.";
    inputRows.push({
      symbolTex: inputSymbols[i],
      detail: guide
    });
  }
  sections.push({
    title: "Input layer (left column in the diagram)",
    rows: inputRows
  });

  let numHidden = network.length - 2;
  for (let layerIdx = 1; layerIdx <= numHidden; layerIdx++) {
    let layer = network[layerIdx];
    let colOrdinal = layerIdx === 1 ? "first" :
        layerIdx === 2 ? "second" :
        layerIdx === 3 ? "third" :
        layerIdx === 4 ? "fourth" :
        layerIdx === 5 ? "fifth" : ordinalSuffix(layerIdx);
    let hiddenRows: LegendRow[] = [];
    for (let i = 0; i < layer.length; i++) {
      let node = layer[i];
      let ordinal = i + 1;
      hiddenRows.push({
        symbolTex: "h_{" + ordinal + "}^{(" + layerIdx + ")}",
        detail: "Neuron " + ordinal + " from the top in the " + colOrdinal +
            " hidden column after inputs (square node id " + node.id +
            " in the diagram)."
      });
    }
    sections.push({
      title: "Hidden layer " + layerIdx + " (" + colOrdinal +
          " column of weighted neurons; " + hiddenActivationSummary + ")",
      rows: hiddenRows
    });
  }

  sections.push({
    title: "Output layer (rightmost column)",
    rows: [{
      symbolTex: "\\hat{y}",
      detail: "Network output after the output activation (" +
          outputActivationSummary + ")."
    }]
  });

  return sections;
}

/** Notation for training / backprop blocks (shown after diagram symbol sections). */
export function buildTrainingNotationLegendSection(
    learningRate: number,
    regularizationKey: string,
    regularizationRate: number): LegendSection {
  let regDetail = regularizationKey === "none" ?
      "No penalty term; the objective is only the mean squared error." :
      "Regularization rate slider; matches lambda in the objective and update equations.";
  return {
    title: "Training notation",
    rows: [
      {
        symbolTex: "\\eta",
        detail: "Learning rate (current value " + learningRate.toFixed(3) +
            ", from the Learning rate control)."
      },
      {
        symbolTex: "\\lambda",
        detail: regDetail +
            (regularizationKey !== "none" ?
                " Current value " + regularizationRate.toFixed(3) + "." :
                "")
      },
      {
        symbolTex: "z",
        detail: "Total input into a neuron (bias plus weighted sum of incoming " +
            "activations). The neuron output is the activation applied to z."
      },
      {
        symbolTex: "\\delta^{(z)}",
        detail: "Derivative of per-example error E with respect to z; " +
            "bias updates use the same accumulated quantity as in the code."
      },
      {
        symbolTex: "E",
        detail: "Per-example squared error (half square of output minus label) " +
            "used in backprop. Training and test loss in the UI are the mean " +
            "over N examples."
      },
      {
        symbolTex: "h_1",
        detail: "In numeric backprop lines, the top hidden neuron in the first " +
            "hidden column (only when the network has at least one hidden layer). " +
            "Values are from the last training example in the most recent Step or " +
            "Play epoch, captured before loss is recomputed over the dataset."
      }
    ]
  };
}
