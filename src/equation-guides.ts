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

interface GuideItem {
  term: string;
  detail: string;
}

function fillDescriptionList(
    container: HTMLElement | null, items: GuideItem[]): void {
  if (container == null) {
    return;
  }
  container.innerHTML = "";
  let dl = document.createElement("dl");
  dl.className = "nn-equation-desc-list";
  for (let i = 0; i < items.length; i++) {
    let it = items[i];
    let dt = document.createElement("dt");
    dt.textContent = it.term;
    let dd = document.createElement("dd");
    dd.textContent = it.detail;
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
  container.appendChild(dl);
}

const FORWARD_GUIDE: GuideItem[] = [
  {
    term: "Layered formula",
    detail: "Each line is one neuron: affine map (bias plus weighted sum of " +
        "incoming values) then the layer activation. Numbers are the live " +
        "weights and biases from the diagram."
  },
  {
    term: "Input symbols (X1, X2, …)",
    detail: "Values from the Features column for one point, in top-to-bottom " +
        "order of input nodes."
  },
  {
    term: "h_j^(l)",
    detail: "Hidden unit j in hidden layer l (column l after inputs), top " +
        "to bottom in the diagram."
  },
  {
    term: "ŷ",
    detail: "Scalar network output after the output activation (regression: " +
        "linear; classification: tanh in this demo)."
  }
];

const OBJECTIVE_GUIDE: GuideItem[] = [
  {
    term: "L (calligraphic in the formula)",
    detail: "Scalar objective minimized during training: average prediction " +
        "error on the training set, plus an optional weight penalty."
  },
  {
    term: "N",
    detail: "Number of training examples used in the average."
  },
  {
    term: "ŷ_i",
    detail: "Network output for training example i (same definition as ŷ in " +
        "the forward pass)."
  },
  {
    term: "y_i",
    detail: "Target label for example i (class or numeric target from the " +
        "dataset)."
  },
  {
    term: "½(ŷ_i − y_i)²",
    detail: "Half squared error for one example; the mean over i is the data " +
        "loss shown as training loss (up to the same formula)."
  },
  {
    term: "λ and ∑_w",
    detail: "When regularization is on: λ is the regularization-rate slider; " +
        "the sum is over all connection weights. L1 adds λ|w|; L2 adds λ·½w² " +
        "per weight (matching the implementation)."
  },
  {
    term: "Gradient of L w.r.t. a weight (data + regularization)",
    detail: "If L = (1/N)Σ_i E_i + λ Σ_w R(w), then for a fixed connection " +
        "weight w, ∂L/∂w = (1/N)Σ_i ∂E_i/∂w + λ R′(w). The first term is what " +
        "backprop builds from the chain rule through the network (prediction → " +
        "layers → that edge). The second term does not pass through ŷ: R depends " +
        "only on w, so its contribution is just λ R′(w). This demo averages " +
        "per-example ∂E/∂w over the mini-batch, then applies a separate " +
        "regularization step—equivalent to adding those two parts of the gradient."
  }
];

const BACKPROP_GUIDE: GuideItem[] = [
  {
    term: "Chain rule (overall)",
    detail: "E depends on ŷ, which depends on a long composition of affine maps " +
        "(z = b + Σ w a) and activations (a = σ(z)). Backprop is the chain rule " +
        "applied in reverse order: start from ∂E/∂ŷ and propagate factors " +
        "∂E/∂z, ∂E/∂w, and ∂E/∂a layer by layer until every weight has a " +
        "∂E/∂w."
  },
  {
    term: "E",
    detail: "Error for a single training example only: E = ½(ŷ − y)². " +
        "Backpropagation computes derivatives of this E (not the full-batch " +
        "average) with respect to weights and activations."
  },
  {
    term: "ŷ and y",
    detail: "Prediction and label for that one example—the same ŷ as after " +
        "forward pass, y from the data point."
  },
  {
    term: "∂E/∂ŷ",
    detail: "Derivative of E with respect to the network output. For this " +
        "square loss it equals ŷ − y; it is the starting signal at the output " +
        "neuron before applying the output activation derivative. This is the " +
        "first factor in the chain: ∂E/∂z_out = (∂E/∂ŷ)(∂ŷ/∂z_out) with " +
        "∂ŷ/∂z_out = σ′(z_out) when ŷ = σ(z_out)."
  },
  {
    term: "δ^(z) and ∂E/∂z",
    detail: "Same quantity: error signal for a neuron’s total input z (bias " +
        "plus weighted sum). In code this is inputDer. It is ∂E/∂z: how much E " +
        "changes if z nudges, after all later layers are folded in by the chain " +
        "rule."
  },
  {
    term: "Chain rule through the activation (σ)",
    detail: "Here a = σ(z). E depends on a, and a depends on z, so by the " +
        "chain rule ∂E/∂z = (∂E/∂a)(da/dz) = (∂E/∂a)σ′(z). That product is " +
        "exactly δ^(z). For a linear output, σ′ = 1 so ∂E/∂z = ∂E/∂a."
  },
  {
    term: "z and a",
    detail: "z is pre-activation input to the neuron; a = σ(z) is the value " +
        "sent along outgoing edges (what the diagram shows on the node)."
  },
  {
    term: "∂E/∂a (before σ′)",
    detail: "How E changes with the neuron’s output a. At the output node, " +
        "∂E/∂a is ∂E/∂ŷ (or merges with it depending on notation). Deeper " +
        "layers receive ∂E/∂a as the sum of backward messages from the next " +
        "column (see the sum formula)."
  },
  {
    term: "Chain rule for one weight",
    detail: "The affine map gives z_to = b + Σ w a_from, so ∂z_to/∂w = a_from " +
        "when w is one of those edges. Hence ∂E/∂w = (∂E/∂z_to)(∂z_to/∂w) = " +
        "δ^(z)_to · a_from (errorDer on that link in code)."
  },
  {
    term: "Chain rule into an earlier activation (the sum)",
    detail: "Each downstream z depends on a_from through z = … + w a_from + …, " +
        "so ∂z/∂a_from = w for that edge. If several units feed from the same " +
        "a_from, E changes through each path: ∂E/∂a_from = Σ_out " +
        "(∂E/∂z_out)(∂z_out/∂a_from) = Σ_out w_out δ^(z)_out. That is the " +
        "displayed sum; it becomes ∂E/∂a for the previous layer’s nodes."
  }
];

const UPDATE_GUIDE: GuideItem[] = [
  {
    term: "w and b",
    detail: "Connection weight and neuron bias updated after a mini-batch of " +
        "examples (each example ran forward and backward and contributed to " +
        "running sums)."
  },
  {
    term: "η (eta)",
    detail: "Learning rate from the UI; scales how far each update moves " +
        "along the negative gradient."
  },
  {
    term: "n_b and the overline",
    detail: "Mini-batch size (batch size control). The bar means average: " +
        "sum of per-example ∂E/∂w or ∂E/∂b over the batch, divided by n_b."
  },
  {
    term: "Second w line (regularization)",
    detail: "If L1 or L2 is enabled: after the gradient step, each weight is " +
        "moved again to shrink the penalty R(w), scaled by η and λ."
  },
  {
    term: "R(w) and R′(w)",
    detail: "Regularization term and its derivative: L2 uses ½w² and R′=w; L1 " +
        "uses |w| with a subgradient in {−1, 0, 1} (0 at w = 0), as in the " +
        "implementation."
  },
  {
    term: "Chain rule: data loss vs. regularization in the update",
    detail: "The averaged term uses ∂E/∂w from backprop—every factor is a chain " +
        "rule through ŷ and the layers. The term λ R′(w) is not chained through " +
        "the network: ∂(λ R(w))/∂w = λ R′(w) because R depends only on w. The " +
        "code applies −(η/n_b)·(sum of ∂E/∂w) and then −ηλ R′(w), which matches " +
        "descending the sum of data-gradient and regularization-gradient on w."
  }
];

/** Populate glossary lists under each equation block. */
export function fillEquationPanelGuides(
    forwardDesc: HTMLElement | null,
    objectiveDesc: HTMLElement | null,
    backpropDesc: HTMLElement | null,
    updateDesc: HTMLElement | null): void {
  fillDescriptionList(forwardDesc, FORWARD_GUIDE);
  fillDescriptionList(objectiveDesc, OBJECTIVE_GUIDE);
  fillDescriptionList(backpropDesc, BACKPROP_GUIDE);
  fillDescriptionList(updateDesc, UPDATE_GUIDE);
}

export const EQUATION_DESC_ELEMENT_IDS: string[] = [
  "nn-equation-forward-desc",
  "nn-equation-objective-desc",
  "nn-equation-backprop-desc",
  "nn-equation-update-desc"
];
