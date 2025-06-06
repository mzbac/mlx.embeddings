import Foundation
import MLX
import MLXNN
import Tokenizers
import XCTest
import mlx_embeddings

struct InstructEmbeddings {
  static func getDetailedInstruct(task: String, query: String) -> String {
    return "Task: \(task)\nQuery: \(query)"
  }
}

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
      let embeddings = modelOutput.textEmbeds
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
      let embeddings = modelOutput.textEmbeds
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

  func testQwen2ModelEmbeddings() async throws {
    print("Loading QWEN2 model container for instruction-based embeddings...")
    let modelContainer = try await mlx_embeddings.loadModelContainer(
      configuration: ModelConfiguration(id: "mlx-community/gte-Qwen2-1.5B-instruct-4bit-dwq")
    )
    print("Model container loaded.")

    // Define task instruction and queries
    let task = "Given a web search query, retrieve relevant passages that answer the query"
    let queries = [
      "how much protein should a female eat",
      "summit define",
    ]

    // Define documents to compare against
    let documents = [
      "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
      "Definition of summit for English Language Learners. : 1 the highest point of a mountain : the top of a mountain. : 2 the highest level. : 3 a meeting or series of meetings between the leaders of two or more governments.",
    ]

    // Prepare all inputs - queries with instructions and plain documents
    let instructedQueries = queries.map {
      InstructEmbeddings.getDetailedInstruct(task: task, query: $0)
    }
    let inputTexts = instructedQueries + documents

    print("Input texts prepared:")
    print("- Instructed queries:")
    for (i, query) in instructedQueries.enumerated() {
      print("  \(i+1). \(query)")
    }
    print("- Documents:")
    for (i, doc) in documents.enumerated() {
      print("  \(i+1). \(doc.prefix(50))...")
    }

    // Generate embeddings for all inputs
    let embeddings: [[Float]] = await modelContainer.perform {
      (model: EmbeddingModel, tokenizer: Tokenizer) -> [[Float]] in

      print("Tokenizing inputs...")
      let tokenizedInputs = inputTexts.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      let maxLength = min(
        tokenizedInputs.reduce(into: 16) { acc, elem in
          acc = max(acc, elem.count)
        },
        8192  // Set maximum context length
      )
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

      print("Running the QWEN2 model...")
      let modelOutput: EmbeddingModelOutput = model(
        paddedInputIds,
        positionIds: nil,
        tokenTypeIds: nil,
        attentionMask: attentionMask
      )
      print("Model execution finished.")

      let embeddings = modelOutput.textEmbeds
      return embeddings.map { $0.asArray(Float.self) }
    }

    print("\n--- Verifying Qwen2 Embedding Results ---")
    
    XCTAssertEqual(embeddings.count, inputTexts.count, 
                   "Should have one embedding for each input text (both instructed queries and documents)")
    
    let expectedDimension = 1536
    if let firstEmbedding = embeddings.first {
      XCTAssertEqual(
        firstEmbedding.count, expectedDimension,
        "Embedding dimension should match the Qwen2 model's expected dimension (\(expectedDimension)).")
      print("Verified embedding dimension: \(firstEmbedding.count)")
      
      let magnitude = sqrt(firstEmbedding.reduce(0) { $0 + $1 * $1 })
      XCTAssertEqual(
        magnitude, 1.0, accuracy: 1e-2,
        "Normalized embedding vector magnitude should be close to 1.0.")
      print(String(format: "Verified normalization (magnitude: %.5f)", magnitude))
      
      if embeddings.count >= 4 {
        let queryEmbedding = embeddings[1]
        let relevantDocEmbedding = embeddings[3]
        
        let dotProduct = zip(queryEmbedding, relevantDocEmbedding).reduce(0) { $0 + $1.0 * $1.1 }
        let queryMagnitude = sqrt(queryEmbedding.reduce(0) { $0 + $1 * $1 })
        let docMagnitude = sqrt(relevantDocEmbedding.reduce(0) { $0 + $1 * $1 })
        let similarity = dotProduct / (queryMagnitude * docMagnitude)
        
        XCTAssertGreaterThan(
          similarity, 0.5,
          "Semantically related query and document should have high similarity score")
        print(String(format: "Verified semantic similarity between related texts: %.5f", similarity))
        
        if embeddings.count >= 5 {
          let unrelatedDocEmbedding = embeddings[3]
          let unrelatedDotProduct = zip(queryEmbedding, unrelatedDocEmbedding).reduce(0) { $0 + $1.0 * $1.1 }
          let unrelatedDocMagnitude = sqrt(unrelatedDocEmbedding.reduce(0) { $0 + $1 * $1 })
          let unrelatedSimilarity = unrelatedDotProduct / (queryMagnitude * unrelatedDocMagnitude)
          
          XCTAssertGreaterThan(
            similarity, unrelatedSimilarity,
            "Semantically related query-doc pair should have higher similarity than unrelated pair")
          print(String(format: "Verified semantic ranking: related similarity (%.5f) > unrelated similarity (%.5f)", 
                      similarity, unrelatedSimilarity))
        }
      }
    } else {
      XCTFail("No embeddings were generated.")
    }
    
    for (i, embedding) in embeddings.enumerated() {
      let nonZeroCount = embedding.filter { abs($0) > 1e-6 }.count
      XCTAssertGreaterThan(
        nonZeroCount, expectedDimension / 10,
        "Embedding should have a significant number of non-zero elements")
      
      let preview = embedding.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", ")
      print("Input \(i): [\(preview), ...] (Dim: \(embedding.count))")
    }
    
    print("Qwen2 embedding test completed successfully.")
  }

  func testQwen3ModelEmbeddings() async throws {
    print("Loading QWEN3 model container for instruction-based embeddings...")
    let modelContainer = try await mlx_embeddings.loadModelContainer(
      configuration: ModelConfiguration(id: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
    )
    print("Model container loaded.")

    // Define task instruction and queries
    let task = "Given a web search query, retrieve relevant passages that answer the query"
    let queries = [
      "how much protein should a female eat",
      "summit define",
    ]

    // Define documents to compare against
    let documents = [
      "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
      "Definition of summit for English Language Learners. : 1 the highest point of a mountain : the top of a mountain. : 2 the highest level. : 3 a meeting or series of meetings between the leaders of two or more governments.",
    ]

    // Prepare all inputs - queries with instructions and plain documents
    let instructedQueries = queries.map {
      InstructEmbeddings.getDetailedInstruct(task: task, query: $0)
    }
    let inputTexts = instructedQueries + documents

    print("Input texts prepared:")
    print("- Instructed queries:")
    for (i, query) in instructedQueries.enumerated() {
      print("  \(i+1). \(query)")
    }
    print("- Documents:")
    for (i, doc) in documents.enumerated() {
      print("  \(i+1). \(doc.prefix(50))...")
    }

    // Generate embeddings for all inputs
    let embeddings: [[Float]] = await modelContainer.perform {
      (model: EmbeddingModel, tokenizer: Tokenizer) -> [[Float]] in

      print("Tokenizing inputs...")
      let tokenizedInputs = inputTexts.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      let maxLength = min(
        tokenizedInputs.reduce(into: 16) { acc, elem in
          acc = max(acc, elem.count)
        },
        8192  // Set maximum context length
      )
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

      print("Running the QWEN3 model...")
      let modelOutput: EmbeddingModelOutput = model(
        paddedInputIds,
        positionIds: nil,
        tokenTypeIds: nil,
        attentionMask: attentionMask
      )
      print("Model execution finished.")

      let embeddings = modelOutput.textEmbeds
      return embeddings.map { $0.asArray(Float.self) }
    }

    print("\n--- Verifying Qwen3 Embedding Results ---")
    
    XCTAssertEqual(embeddings.count, inputTexts.count, 
                   "Should have one embedding for each input text (both instructed queries and documents)")
    
    let expectedDimension = 1024
    if let firstEmbedding = embeddings.first {
      XCTAssertEqual(
        firstEmbedding.count, expectedDimension,
        "Embedding dimension should match the Qwen3 model's expected dimension (\(expectedDimension)).")
      print("Verified embedding dimension: \(firstEmbedding.count)")
      
      let magnitude = sqrt(firstEmbedding.reduce(0) { $0 + $1 * $1 })
      XCTAssertEqual(
        magnitude, 1.0, accuracy: 1e-2,
        "Normalized embedding vector magnitude should be close to 1.0.")
      print(String(format: "Verified normalization (magnitude: %.5f)", magnitude))
      
      if embeddings.count >= 4 {
        let queryEmbedding = embeddings[0]
        let relevantDocEmbedding = embeddings[2]
        
        let dotProduct = zip(queryEmbedding, relevantDocEmbedding).reduce(0) { $0 + $1.0 * $1.1 }
        let queryMagnitude = sqrt(queryEmbedding.reduce(0) { $0 + $1 * $1 })
        let docMagnitude = sqrt(relevantDocEmbedding.reduce(0) { $0 + $1 * $1 })
        let similarity = dotProduct / (queryMagnitude * docMagnitude)
        
        XCTAssertGreaterThan(
          similarity, 0.5,
          "Semantically related query and document should have high similarity score")
        print(String(format: "Verified semantic similarity between related texts: %.5f", similarity))
        
        if embeddings.count >= 5 {
          let unrelatedDocEmbedding = embeddings[3]
          let unrelatedDotProduct = zip(queryEmbedding, unrelatedDocEmbedding).reduce(0) { $0 + $1.0 * $1.1 }
          let unrelatedDocMagnitude = sqrt(unrelatedDocEmbedding.reduce(0) { $0 + $1 * $1 })
          let unrelatedSimilarity = unrelatedDotProduct / (queryMagnitude * unrelatedDocMagnitude)
          
          XCTAssertGreaterThan(
            similarity, unrelatedSimilarity,
            "Semantically related query-doc pair should have higher similarity than unrelated pair")
          print(String(format: "Verified semantic ranking: related similarity (%.5f) > unrelated similarity (%.5f)", 
                      similarity, unrelatedSimilarity))
        }
      }
    } else {
      XCTFail("No embeddings were generated.")
    }
    
    for (i, embedding) in embeddings.enumerated() {
      let nonZeroCount = embedding.filter { abs($0) > 1e-6 }.count
      XCTAssertGreaterThan(
        nonZeroCount, expectedDimension / 10,
        "Embedding should have a significant number of non-zero elements")
      
      let preview = embedding.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", ")
      print("Input \(i): [\(preview), ...] (Dim: \(embedding.count))")
    }
    
    print("Qwen3 embedding test completed successfully.")
  }
}
