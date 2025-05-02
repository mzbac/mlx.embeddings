import Foundation
import MLX
import MLXNN
import Tokenizers
import XCTest
import mlx_embeddings

class BertModelTests: XCTestCase {

  func testBertModelEmbeddings() async throws {

    print("Loading BERT model container...")
    let modelContainer = try await mlx_embeddings.loadModelContainer(
      configuration: ModelConfiguration(id: "TaylorAI/bge-micro"))
    print("Model container loaded.")

    let inputs = [
      "This is the first sentence.",
      "Here is another sentence, slightly longer.",
      "A third short one.",
    ]
    print("Input texts: \(inputs)")

    let embeddings: [[Float]] = await modelContainer.perform {
      (model: EmbeddingModel, tokenizer: Tokenizer) -> [[Float]] in

      print("Tokenizing inputs...")
      let tokenizedInputs = inputs.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      let maxLength = tokenizedInputs.reduce(into: 16) { acc, elem in
        acc = max(acc, elem.count)
      }
      print("Padding inputs to max length: \(maxLength)")

      let padTokenId = tokenizer.eosTokenId ?? 0
      let paddedInputIds = MLX.stacked(
        tokenizedInputs.map { elem -> MLXArray in
          let paddingCount = maxLength - elem.count
          let paddedArray = elem + Array(repeating: padTokenId, count: paddingCount)
          return MLXArray(paddedArray)
        }
      )

      let attentionMask = paddedInputIds .!= MLXArray(padTokenId)
      print("Created attention mask.")

      let tokenTypeIds = MLXArray.zeros(like: paddedInputIds)
      print("Created token type IDs (all zeros).")

      print("Running the BERT model...")
      let modelOutput: EmbeddingModelOutput = model(
        paddedInputIds,
        positionIds: nil,
        tokenTypeIds: tokenTypeIds,
        attentionMask: attentionMask
      )
      print("Model execution finished.")
      let embeddings = modelOutput.textEmbeds!
      return embeddings.map { $0.asArray(Float.self) }
    }

    print("\n--- Verifying Embedding Results ---")
    XCTAssertEqual(embeddings.count, inputs.count, "Should have one embedding per input sentence.")

    let expectedDimension = 384
    if let firstEmbedding = embeddings.first {
      XCTAssertEqual(
        firstEmbedding.count, expectedDimension,
        "Embedding dimension should match the model's expected dimension (\(expectedDimension)).")
      print("Verified embedding dimension: \(firstEmbedding.count)")

      let magnitude = sqrt(firstEmbedding.reduce(0) { $0 + $1 * $1 })
      XCTAssertEqual(
        magnitude, 1.0, accuracy: 1e-5,
        "Normalized embedding vector magnitude should be close to 1.0.")
      print(String(format: "Verified normalization (magnitude: %.5f)", magnitude))

    } else {
      XCTFail("No embeddings were generated.")
    }

    for (i, embedding) in embeddings.enumerated() {
      let preview = embedding.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", ")
      print("Input \(i): [\(preview), ...] (Dim: \(embedding.count))")
    }
    print("Test completed successfully.")
  }

  func testDistilBertModelEmbeddings() async throws {

    print("Loading DISTILBERT model container...")
    let modelContainer = try await mlx_embeddings.loadModelContainer(
      configuration: ModelConfiguration(id: "distilbert/distilbert-base-uncased")
    )
    print("Model container loaded.")

    let inputs = [
      "This is the first sentence.",
      "Here is another sentence, slightly longer.",
      "A third short one.",
    ]
    print("Input texts: \(inputs)")

    let embeddings: [[Float]] = await modelContainer.perform {
      (model: EmbeddingModel, tokenizer: Tokenizer) -> [[Float]] in

      print("Tokenizing inputs...")
      let tokenizedInputs = inputs.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      let maxLength = tokenizedInputs.reduce(into: 16) { acc, elem in
        acc = max(acc, elem.count)
      }
      print("Padding inputs to max length: \(maxLength)")

      let padTokenId = tokenizer.eosTokenId ?? 0
      let paddedInputIds = MLX.stacked(
        tokenizedInputs.map { elem -> MLXArray in
          let paddingCount = maxLength - elem.count
          let paddedArray = elem + Array(repeating: padTokenId, count: paddingCount)
          return MLXArray(paddedArray)
        }
      )

      let attentionMask = paddedInputIds .!= MLXArray(padTokenId)
      print("Created attention mask.")

      let tokenTypeIds = MLXArray.zeros(like: paddedInputIds)
      print("Created token type IDs (all zeros).")

      print("Running the BERT model...")
      let modelOutput: EmbeddingModelOutput = model(
        paddedInputIds,
        positionIds: nil,
        tokenTypeIds: tokenTypeIds,
        attentionMask: attentionMask
      )
      print("Model execution finished.")
      let embeddings = modelOutput.textEmbeds!
      return embeddings.map { $0.asArray(Float.self) }
    }

    print("\n--- Verifying Embedding Results ---")
    XCTAssertEqual(embeddings.count, inputs.count, "Should have one embedding per input sentence.")

    let expectedDimension = 768
    if let firstEmbedding = embeddings.first {
      XCTAssertEqual(
        firstEmbedding.count, expectedDimension,
        "Embedding dimension should match the model's expected dimension (\(expectedDimension)).")
      print("Verified embedding dimension: \(firstEmbedding.count)")

      let magnitude = sqrt(firstEmbedding.reduce(0) { $0 + $1 * $1 })
      XCTAssertEqual(
        magnitude, 1.0, accuracy: 1e-5,
        "Normalized embedding vector magnitude should be close to 1.0.")
      print(String(format: "Verified normalization (magnitude: %.5f)", magnitude))

    } else {
      XCTFail("No embeddings were generated.")
    }

    for (i, embedding) in embeddings.enumerated() {
      let preview = embedding.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", ")
      print("Input \(i): [\(preview), ...] (Dim: \(embedding.count))")
    }
    print("Test completed successfully.")
  }
}
