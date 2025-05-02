import Foundation
import MLX
import MLXFast
import MLXNN

extension MLXArray {
  public static func arange(_ size: Int) -> MLXArray {
    return MLXArray(Array(0..<size))
  }
}

private class BertEmbedding: Module {

  let typeVocabularySize: Int
  @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
  @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
  @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding?
  @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding

  init(_ config: BertConfiguration) {
    typeVocabularySize = config.type_vocab_size
    _wordEmbeddings.wrappedValue = Embedding(
      embeddingCount: config.vocab_size, dimensions: config.hidden_size)
    _norm.wrappedValue = LayerNorm(
      dimensions: config.hidden_size, eps: config.layer_norm_eps)
    if config.type_vocab_size > 0 {
      _tokenTypeEmbeddings.wrappedValue = Embedding(
        embeddingCount: config.type_vocab_size,
        dimensions: config.hidden_size)
    }
    _positionEmbeddings.wrappedValue = Embedding(
      embeddingCount: config.max_position_embeddings,
      dimensions: config.hidden_size)

  }

  func callAsFunction(
    inputIds: MLXArray,
    positionIds: MLXArray? = nil,
    tokenTypeIds: MLXArray? = nil
  ) -> MLXArray {
    let posIds = positionIds ?? broadcast(MLXArray.arange(inputIds.dim(1)), to: inputIds.shape)
    let words = wordEmbeddings(inputIds) + positionEmbeddings(posIds)
    if let tokenTypeIds, let tokenTypeEmbeddings {
      words += tokenTypeEmbeddings(tokenTypeIds)
    }
    return norm(words)
  }
}

private class BertSelfAttention: Module {
  let numAttentionHeads: Int
  let attentionHeadSize: Int
  let allHeadSize: Int

  @ModuleInfo(key: "query") var query: Linear
  @ModuleInfo(key: "key") var key: Linear
  @ModuleInfo(key: "value") var value: Linear

  init(_ config: BertConfiguration) {
    numAttentionHeads = config.num_attention_heads
    attentionHeadSize = config.hidden_size / config.num_attention_heads
    allHeadSize = numAttentionHeads * attentionHeadSize

    _query.wrappedValue = Linear(config.hidden_size, allHeadSize)
    _key.wrappedValue = Linear(config.hidden_size, allHeadSize)
    _value.wrappedValue = Linear(config.hidden_size, allHeadSize)
  }

  func transposeForScores(_ x: MLXArray) -> MLXArray {
    let batchSize = x.dim(0)
    let seqLength = x.dim(1)

    let newShape = [batchSize, seqLength, numAttentionHeads, attentionHeadSize]

    let reshaped = reshaped(x, newShape)
    return reshaped.transposed(0, 2, 1, 3)
  }

  func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let mixedQueryLayer = query(hiddenStates)
    let mixedKeyLayer = key(hiddenStates)
    let mixedValueLayer = value(hiddenStates)

    let queryLayer = transposeForScores(mixedQueryLayer)
    let keyLayer = transposeForScores(mixedKeyLayer)
    let valueLayer = transposeForScores(mixedValueLayer)

    var attentionScores = matmul(queryLayer, keyLayer.transposed(0, 1, 3, 2))
    attentionScores = attentionScores / MLXArray(sqrt(Float(attentionHeadSize)))

    if let mask = mask {
      attentionScores = attentionScores + mask
    }

    let attentionProbs = softmax(attentionScores, axis: -1)

    var contextLayer = matmul(attentionProbs, valueLayer)

    contextLayer = contextLayer.transposed(0, 2, 1, 3)

    let outputShape = [contextLayer.dim(0), contextLayer.dim(1), allHeadSize]
    return contextLayer.reshaped(outputShape)
  }
}

private class BertSelfOutput: Module {
  @ModuleInfo(key: "dense") var dense: Linear
  @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

  init(_ config: BertConfiguration) {
    _dense.wrappedValue = Linear(config.hidden_size, config.hidden_size)
    _layerNorm.wrappedValue = LayerNorm(dimensions: config.hidden_size, eps: config.layer_norm_eps)
  }

  func callAsFunction(_ hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray {
    let hiddenStates = dense(hiddenStates)
    return layerNorm(hiddenStates + inputTensor)
  }
}

private class BertAttention: Module {
  @ModuleInfo(key: "self") var selfAttention: BertSelfAttention
  @ModuleInfo(key: "output") var selfOutput: BertSelfOutput

  init(_ config: BertConfiguration) {
    _selfAttention.wrappedValue = BertSelfAttention(config)
    _selfOutput.wrappedValue = BertSelfOutput(config)
  }

  func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let selfOutputs = selfAttention(inputs, mask: mask)
    let attentionOutput = selfOutput(selfOutputs, inputTensor: inputs)
    return attentionOutput
  }
}

private class BertIntermediate: Module {
  @ModuleInfo(key: "dense") var dense: Linear

  init(_ config: BertConfiguration) {
    _dense.wrappedValue = Linear(config.hidden_size, config.intermediate_size)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    let hiddenStates = dense(hiddenStates)
    return gelu(hiddenStates)
  }
}

private class BertOutput: Module {
  @ModuleInfo(key: "dense") var dense: Linear
  @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

  init(_ config: BertConfiguration) {
    _dense.wrappedValue = Linear(config.intermediate_size, config.hidden_size)
    _layerNorm.wrappedValue = LayerNorm(dimensions: config.hidden_size, eps: config.layer_norm_eps)
  }

  func callAsFunction(_ hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray {
    var hiddenStates = dense(hiddenStates)
    hiddenStates = hiddenStates + inputTensor
    return layerNorm(hiddenStates)
  }
}

private class TransformerBlock: Module {
  @ModuleInfo(key: "attention") var attention: BertAttention
  @ModuleInfo(key: "intermediate") var intermediate: BertIntermediate
  @ModuleInfo(key: "output") var output: BertOutput

  init(_ config: BertConfiguration) {
    _attention.wrappedValue = BertAttention(config)
    _intermediate.wrappedValue = BertIntermediate(config)
    _output.wrappedValue = BertOutput(config)
  }

  func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let attentionOut = attention(inputs, mask: mask)
    let intermediateOutput = intermediate(attentionOut)
    let layerOutput = output(intermediateOutput, inputTensor: attentionOut)
    return layerOutput
  }
}

private class BertEncoder: Module {
  @ModuleInfo(key: "layer") fileprivate var layer: [TransformerBlock]
  init(_ config: BertConfiguration) {
    precondition(config.vocab_size > 0)
    _layer.wrappedValue = (0..<config.num_hidden_layers).map { _ in TransformerBlock(config) }
  }
  func callAsFunction(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
    var outputs = inputs
    for l in layer {
      outputs = l(outputs, mask: attentionMask)
    }
    return outputs
  }
}

private class BertPooler: Module {
  @ModuleInfo(key: "dense") var dense: Linear

  init(_ config: BertConfiguration) {
    _dense.wrappedValue = Linear(config.hidden_size, config.hidden_size)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    let firstTokenTensor = hiddenStates[0..., 0]
    return tanh(dense(firstTokenTensor))
  }
}

public class BertModel: Module, EmbeddingModel {
  @ModuleInfo(key: "embeddings") fileprivate var embedding: BertEmbedding
  @ModuleInfo(key: "pooler") fileprivate var pooler: BertPooler
  fileprivate let encoder: BertEncoder

  public init(
    _ config: BertConfiguration
  ) {
    encoder = BertEncoder(config)
    _embedding.wrappedValue = BertEmbedding(config)
    _pooler.wrappedValue = BertPooler(config)

  }

  fileprivate func getExtendedAttentionMask(_ attentionMask: MLXArray) -> MLXArray {
    if attentionMask.ndim == 3 {
      let extended = attentionMask.expandedDimensions(axes: [1])
      return (1.0 - extended) * -10000.0
    } else if attentionMask.ndim == 2 {
      let extended = attentionMask.expandedDimensions(axes: [1, 2])
      return (1.0 - extended) * -10000.0
    } else {
      fatalError("Wrong shape for attention_mask (shape \(attentionMask.shape))")
    }
  }

  public func callAsFunction(
    _ inputIds: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
    attentionMask: MLXArray? = nil
  )
    -> EmbeddingModelOutput
  {
    let (batchSize, seqLen) = (inputIds.dim(0), inputIds.dim(1))
    let embeddingOutput = embedding(inputIds: inputIds, tokenTypeIds: tokenTypeIds)
    let _mask = attentionMask ?? MLXArray.ones([batchSize, seqLen])

    let mask = getExtendedAttentionMask(_mask)

    let encoderOutput = encoder(
      embeddingOutput,
      attentionMask: mask)

    let poolerOutput = self.pooler(encoderOutput)
    var text_embeds = meanPooling(
      lastHiddenState: encoderOutput,
      attentionMask: _mask)
    text_embeds = normalizeEmbeddings(text_embeds)

    return EmbeddingModelOutput(
      hiddenStates: encoderOutput,
      poolerOutput: poolerOutput,
      textEmbeds: text_embeds)
  }

  public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitizedWeights = [String: MLXArray]()
    for (key, value) in weights {
      if key.contains("position_ids") {
        continue
      } else {
        let sanitizedKey = key.replacingOccurrences(of: "bert.", with: "")
        sanitizedWeights[sanitizedKey] = value
      }
    }
    return sanitizedWeights
  }
}

public struct BertConfiguration: Decodable, Sendable {
  var layer_norm_eps: Float = 1e-12
  var max_trained_positions: Int = 2048
  var hidden_size: Int = 768
  var num_attention_heads: Int = 12
  var intermediate_size: Int = 3072
  var num_hidden_layers: Int = 12
  var type_vocab_size: Int = 2
  var vocab_size: Int = 30528
  var max_position_embeddings: Int = 0
  var model_type: String

  enum CodingKeys: String, CodingKey {
    case layer_norm_eps
    case max_trained_positions
    case vocab_size
    case max_position_embeddings
    case model_type
  }

  enum BertCodingKeys: String, CodingKey {
    case hidden_size
    case num_attention_heads
    case intermediate_size
    case num_hidden_layers
    case type_vocab_size
  }

  public init(from decoder: Decoder) throws {
    let container: KeyedDecodingContainer<CodingKeys> =
      try decoder.container(
        keyedBy: CodingKeys.self)
    layer_norm_eps =
      try container.decodeIfPresent(
        Float.self,
        forKey: CodingKeys.layer_norm_eps)
      ?? 1e-12
    max_trained_positions =
      try container.decodeIfPresent(
        Int.self,
        forKey: CodingKeys.max_trained_positions) ?? 2048
    vocab_size =
      try container.decodeIfPresent(
        Int.self,
        forKey: CodingKeys.vocab_size)
      ?? 30528
    max_position_embeddings =
      try container.decodeIfPresent(
        Int.self,
        forKey: CodingKeys.max_position_embeddings) ?? 0
    model_type = try container.decode(String.self, forKey: CodingKeys.model_type)

    let bertConfig: KeyedDecodingContainer<BertCodingKeys> = try decoder.container(
      keyedBy: BertCodingKeys.self)

    hidden_size =
      try bertConfig.decodeIfPresent(
        Int.self,
        forKey: BertCodingKeys.hidden_size) ?? 768
    num_attention_heads =
      try bertConfig.decodeIfPresent(
        Int.self,
        forKey: BertCodingKeys.num_attention_heads) ?? 12
    intermediate_size =
      try bertConfig.decodeIfPresent(
        Int.self, forKey: BertCodingKeys.intermediate_size)
      ?? 3072
    num_hidden_layers =
      try bertConfig.decodeIfPresent(
        Int.self,
        forKey: BertCodingKeys.num_hidden_layers) ?? 12
    type_vocab_size =
      try bertConfig.decodeIfPresent(
        Int.self,
        forKey: BertCodingKeys.type_vocab_size)
      ?? 2
  }
}
