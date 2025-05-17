import Foundation
import MLX
import MLXFast
import MLXLinalg
import MLXNN

public func meanPooling(lastHiddenState: MLXArray, attentionMask: MLXArray) -> MLXArray {
  let expandedMask = attentionMask.expandedDimensions(axes: [-1])

  let broadcastMask = broadcast(expandedMask, to: lastHiddenState.shape).asType(.float32)
  let sumHiddenState = sum(lastHiddenState * broadcastMask, axes: [1])
  let sumMask = sum(broadcastMask, axes: [1])
  let safeSumMask = MLX.maximum(sumMask, MLXArray(1e-9))
  return sumHiddenState / safeSumMask
}

public func normalizeEmbeddings(_ embeddings: MLXArray) -> MLXArray {
  let normValue = norm(embeddings, ord: 2, axis: -1, keepDims: true)
  let safeNormValue = MLX.maximum(normValue, MLXArray(1e-9))
  return embeddings / safeNormValue
}

public func lastTokenPooling(lastHiddenState: MLXArray, attentionMask: MLXArray) -> MLXArray {
    let sequenceLengths = sum(attentionMask, axes: [1]) - 1
    let batchSize = lastHiddenState.shape[0]
    let lastTokenIndices = maximum(sequenceLengths, MLXArray(0))
    
    let batchIndices = MLXArray.arange(batchSize)
    return lastHiddenState[batchIndices, lastTokenIndices]
}
