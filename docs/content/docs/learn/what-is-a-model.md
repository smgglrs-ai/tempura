+++
title = "0. What is a Model?"
description = "What AI models actually are, what they aren't, and how they relate to the math we'll learn."
weight = 5
template = "docs/page.html"

[extra]
toc = true
math = true
+++

## What you already know

You've used AI. Maybe you've asked ChatGPT a question, or had your phone autocomplete a sentence, or seen AI-generated images. Something goes in (your question), something comes out (an answer), and it often feels like the machine *understands*.

This course will show you what actually happens inside. No magic, no hand-waving — just math that you can follow with a pencil and paper. But before we start with the math, let's establish what we're building toward.

## A model is a recipe for turning input into output

That's it. A model is a specific set of instructions — expressed as math operations — that transforms input data into output data.

When you type "What is the capital of France?" into a language model, here's what actually happens:

1. Your text is split into small pieces (tokens)
2. Each piece is converted into a list of numbers
3. Those numbers are multiplied by other numbers, added together, and passed through mathematical functions — millions of times
4. The result is a list of numbers representing probabilities for what the next word should be
5. A word is chosen based on those probabilities
6. Steps 3-5 repeat for each word of the response

No understanding. No consciousness. No database lookup. Just arithmetic — addition, multiplication, and a few other operations — applied to enormous tables of numbers.

## Where do the numbers come from?

The tables of numbers (called **weights** or **parameters**) are the model's knowledge. A 7 billion parameter model has 7 billion numbers, arranged in specific patterns, that collectively encode the ability to produce coherent text.

These numbers aren't programmed by humans. They're **learned** — adjusted automatically by a process that involves showing the model billions of examples of text and gradually tuning the numbers to make better predictions. This process is called **training**.

Once training is done, the numbers are fixed. Using the model (called **inference**) means running the math with those fixed numbers on new input. This is what tempura does — it runs the math.

## The recipe analogy

Think of a model like a very elaborate recipe:

- **The recipe** (the architecture): "take the input, multiply by table A, add table B, apply function C, repeat 32 times..." This is the same for all models of the same type.
- **The ingredients** (the weights/parameters): the specific numbers in tables A, B, etc. Different models have different numbers, learned from different data.
- **The kitchen** (the inference engine): the software and hardware that executes the recipe. This is what tempura is — a kitchen.

A recipe doesn't understand food. It doesn't taste anything. But if you follow it precisely, you get a predictable result. Similarly, a language model doesn't understand language — but if you run the math precisely, you get a response that often *looks like* understanding.

## What a model is NOT

**Not a database.** The model doesn't store facts in a lookup table. "The capital of France is Paris" isn't saved as a string anywhere. Instead, the weight values are arranged so that when the math is run on the input "What is the capital of France?", the output probabilities are highest for tokens that form "Paris."

**Not a search engine.** The model doesn't look anything up on the internet (unless explicitly connected to one). Everything it generates comes from the patterns encoded in its weights during training.

**Not a thinking being.** The model doesn't have goals, beliefs, or understanding. It predicts the most likely next word, over and over. The appearance of reasoning emerges from patterns in the training data, not from any internal deliberation.

**Not deterministic by default.** Most models include a random element in choosing each word (see [Probability](../probability/)). The same question can get different answers. tempura offers the option to make this deterministic — same input, same seed, same output, every time.

## How big are these models?

To give you a sense of scale:

| Model | Parameters | Weight file size | What it can do |
|-------|-----------|-----------------|---------------|
| Tiny (for testing) | 100K | 400 KB | Barely coherent text |
| GPT-2 Small | 117M | 500 MB | Short coherent text |
| LLaMA 7B | 7B | 14 GB | Conversational AI |
| LLaMA 70B | 70B | 140 GB | Advanced reasoning |
| GPT-4 class | ~1T (estimated) | ~2 TB | State of the art |

The numbers grow, but the math stays the same — just more of it. A 70B model does the same operations as a 7B model, just with wider matrices and more layers.

## What we'll learn in this course

Over the next 9 chapters, we'll build up the math behind these models piece by piece:

1. **[Vectors](../vectors/)** — how to represent information as numbers
2. **[Matrices](../matrices/)** — how to transform that information
3. **[Functions](../functions/)** — why we need non-linear operations
4. **[Probability](../probability/)** — how models make choices
5. **[Learning](../learning/)** — how the weight numbers are found
6. **[Neural Networks](../neural-networks/)** — how layers combine into a model
7. **[Words as Numbers](../words-as-numbers/)** — how text becomes math
8. **[The Transformer](../the-transformer/)** — the specific architecture used by modern AI
9. **[Inference](../inference/)** — what happens when you use a model

By the end, there will be no mystery left. You'll understand every operation from input to output.

## One more thing: why tempura?

Most inference engines are black boxes — text goes in, text comes out, and you have to trust that the math was done correctly.

tempura is different. It **proves** that certain operations are safe (using a tool called [Kani](../../foundations/bounded-model-checking/)), **compares** different implementations to verify they produce the same results (using [DualPipeline](../../foundations/dual-pipeline/)), and **records** what happened at every step (using [InferenceTrace](../../trace-tooling/)). The [Foundations](../../foundations/) section covers the verification in depth — but first, let's learn the math.

**Next:** [Vectors](../vectors/) — the first building block.
