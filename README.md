# MLX Embeddings Swift Library

[![Swift Version](https://img.shields.io/badge/Swift-6.0+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://developer.apple.com/macos)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Swift library built with [MLX Swift](https://github.com/ml-explore/mlx-swift) for creating and using embedding models. This library provides a modern Swift interface for generating, manipulating, and utilizing embeddings in machine learning applications on Apple Silicon.

## Overview

MLX Embeddings aims to simplify the process of working with text embeddings in Swift. It leverages the performance of Apple Silicon via the MLX framework, offering efficient computation for various embedding tasks.

## Features

*   Load pre-trained embedding models.
*   Generate embeddings for single texts or batches.

## Limitations

- **Currently only BERT-based models are supported.** Support for other architectures may be added in the future.

## Requirements

*   macOS 14.0+
*   Xcode 16.0+
*   Swift 6.0+
*   An Apple Silicon Mac (M1, M2, M3, M4 series)

## Installation

Add MLX Embeddings as a dependency to your `Package.swift` file:

```swift
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "YourProjectName",
    platforms: [
        .macOS(.v14) // Minimum macOS version
    ],
    dependencies: [
        .package(url: "https://github.com/mzbac/mlx.embeddings.git", from: "0.1.0")
    ],
    targets: [
        .executableTarget(
            name: "YourProjectName",
            dependencies: [
                .product(name: "mlx_embeddings", package: "mlx_embeddings"),
            ]
        )
    ]
)
```

Then run `swift build`.

## Usage Example

```swift
import Foundation
import MLX
import Tokenizers
import mlx_embeddings


Task {
  do {
    let inputs = [
      "This is the first sentence.",
      "Here is another sentence, slightly longer.",
      "A third short one.",
    ]

    let container = try await loadModelContainer(
      configuration: ModelConfiguration(id: "TaylorAI/bge-micro")
    )

    _ = await container.perform { model, tokenizer in
      let tokenized = inputs.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }
      let maxLength = tokenized.reduce(into: 16) { acc, elem in
        acc = max(acc, elem.count)
      }
      let padId = tokenizer.eosTokenId ?? 0

      let padded = stacked(
        tokenized.map { elem in
          MLXArray(
            elem
              + Array(
                repeating: padId,
                count: maxLength - elem.count))
        })
      let attentionMask = padded .!= MLXArray(padId)
      let tokenTypeIds = MLXArray.zeros(like: padded)

      let output = model(
        padded,
        positionIds: nil,
        tokenTypeIds: tokenTypeIds,
        attentionMask: attentionMask
      )

      let embeddings = output.textEmbeds!

      // simple cosine similarity
      func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let dot_product = (a * b).sum()
        let norm_a = MLX.sqrt((a * a).sum())
        let norm_b = MLX.sqrt((b * b).sum())
        return (dot_product / (norm_a * norm_b)).item()
      }
      print("Similarity[0–1]:", cosine(embeddings[0], embeddings[1]))

    }

  } catch {
    print("Error:", error)
  }
}

dispatchMain()
```

## Acknowledgements

- The model implementation was ported from [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings).
- Utility classes and supporting code were adapted from [mlx-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples/tree/main/Libraries/Embedders).