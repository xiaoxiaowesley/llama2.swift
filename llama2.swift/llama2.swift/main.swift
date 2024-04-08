//
//  main.swift
//  llama2.swift
//
//  Created by xiaoxiang's m1 mbp on 2024/4/8.
//

import Foundation

// ----------------------------------------------------------------------------
// Transformer model
struct Config {
    var dim: Int // transformer dimension
    var hidden_dim: Int // for ffn layers
    var n_layers: Int // number of layers
    var n_heads: Int // number of query heads
    var n_kv_heads: Int // number of key/value heads (can be < query heads because of multiquery)
    var vocab_size: Int // vocabulary size, usually 256 (byte-level)
    var seq_len: Int // max sequence length
}

struct TransformerWeights {
    // token embedding table
    var token_embedding_table: [Float]    // (vocab_size, dim)
    // weights for rmsnorms
    var rms_att_weight: [Float] // (layer, dim) rmsnorm weights
    var rms_ffn_weight: [Float] // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    var wq: [Float] // (layer, dim, n_heads * head_size)
    var wk: [Float] // (layer, dim, n_kv_heads * head_size)
    var wv: [Float] // (layer, dim, n_kv_heads * head_size)
    var wo: [Float] // (layer, n_heads * head_size, dim)
    // weights for ffn
    var w1: [Float] // (layer, hidden_dim, dim)
    var w2: [Float] // (layer, dim, hidden_dim)
    var w3: [Float] // (layer, hidden_dim, dim)
    // final rmsnorm
    var rms_final_weight: [Float] // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    var wcls: [Float]?
}

struct RunState {
    // current wave of activations
    var x: [Float] // activation at current time stamp (dim,)
    var xb: [Float] // same, but inside a residual branch (dim,)
    var xb2: [Float] // an additional buffer just for convenience (dim,)
    var hb: [Float] // buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: [Float] // buffer for hidden dimension in the ffn (hidden_dim,)
    var q: [Float] // query (dim,)
    var k: [Float] // key (dim,)
    var v: [Float] // value (dim,)
    var att: [Float] // buffer for scores/attention values (n_heads, seq_len)
    var logits: [Float] // output logits
    // kv cache
    var key_cache: [Float]   // (layer, seq_len, dim)
    var value_cache: [Float] // (layer, seq_len, dim)
}

struct Transformer {
    var config: Config // the hyperparameters of the architecture (the blueprint)
    var weights: TransformerWeights // the weights of the model
    var state: RunState // buffers for the "wave" of activations in the forward pass
    var fd: Int32 // file descriptor for memory mapping
    var data: UnsafeMutablePointer<Float>? // memory mapped data pointer
    var fileSize: Int // size of the checkpoint file in bytes
}

func mallocRunState(s: inout RunState, p: Config) {
    let kvDim = (p.dim * p.n_kv_heads) / p.n_heads
    s.x = Array(repeating: 0.0, count: p.dim)
    s.xb = Array(repeating: 0.0, count: p.dim)
    s.xb2 = Array(repeating: 0.0, count: p.dim)
    s.hb = Array(repeating: 0.0, count: p.hidden_dim)
    s.hb2 = Array(repeating: 0.0, count: p.hidden_dim)
    s.q = Array(repeating: 0.0, count: p.dim)
    s.key_cache = Array(repeating: 0.0, count: p.n_layers * p.seq_len * kvDim)
    s.value_cache = Array(repeating: 0.0, count: p.n_layers * p.seq_len * kvDim)
    s.att = Array(repeating: 0.0, count: p.n_heads * p.seq_len)
    s.logits = Array(repeating: 0.0, count: p.vocab_size)

    // ensure all mallocs went fine
    if s.x.isEmpty || s.xb.isEmpty || s.xb2.isEmpty || s.hb.isEmpty || s.hb2.isEmpty || s.q.isEmpty
        || s.key_cache.isEmpty || s.value_cache.isEmpty || s.att.isEmpty || s.logits.isEmpty {
        print("malloc failed!")
        exit(EXIT_FAILURE)
    }    
}

func freeRunState(s: inout RunState) {
    s.x.removeAll()
    s.xb.removeAll()
    s.xb2.removeAll()
    s.hb.removeAll()
    s.hb2.removeAll()
    s.q.removeAll()
    s.att.removeAll()
    s.logits.removeAll()
    s.key_cache.removeAll()
    s.value_cache.removeAll()
}

func memoryMapWeights(w: inout TransformerWeights, p: Config, ptr: inout [Float], sharedWeights: Bool) {
    let headSize = p.dim / p.n_heads
    let nLayers = p.n_layers
    let vocabSizeDim = p.vocab_size * p.dim
    let nLayersDim = nLayers * p.dim
    let nLayersDimHeads = nLayers * p.dim * (p.n_heads * headSize)
    let nLayersDimKVHeads = nLayers * p.dim * (p.n_kv_heads * headSize)
    let nLayersHeadsDim = nLayers * (p.n_heads * headSize) * p.dim
    let nLayersDimHiddenDim = nLayers * p.dim * p.hidden_dim
    let nLayersHiddenDimDim = nLayers * p.hidden_dim * p.dim
    let seqLenHeadSize = p.seq_len * headSize / 2

    w.token_embedding_table = Array(ptr[0..<vocabSizeDim])
    ptr.removeFirst(vocabSizeDim)

    w.rms_att_weight = Array(ptr[0..<nLayersDim])
    ptr.removeFirst(nLayersDim)

    w.wq = Array(ptr[0..<nLayersDimHeads])
    ptr.removeFirst(nLayersDimHeads)

    w.wk = Array(ptr[0..<nLayersDimKVHeads])
    ptr.removeFirst(nLayersDimKVHeads)

    w.wv = Array(ptr[0..<nLayersDimKVHeads])
    ptr.removeFirst(nLayersDimKVHeads)

    w.wo = Array(ptr[0..<nLayersHeadsDim])
    ptr.removeFirst(nLayersHeadsDim)

    w.rms_ffn_weight = Array(ptr[0..<nLayersDim])
    ptr.removeFirst(nLayersDim)

    w.w1 = Array(ptr[0..<nLayersDimHiddenDim])
    ptr.removeFirst(nLayersDimHiddenDim)

    w.w2 = Array(ptr[0..<nLayersHiddenDimDim])
    ptr.removeFirst(nLayersHiddenDimDim)

    w.w3 = Array(ptr[0..<nLayersDimHiddenDim])
    ptr.removeFirst(nLayersDimHiddenDim)

    w.rms_final_weight = Array(ptr[0..<p.dim])
    ptr.removeFirst(p.dim)

    ptr.removeFirst(seqLenHeadSize) // skip what used to be freq_cis_real (for RoPE)
    ptr.removeFirst(seqLenHeadSize) // skip what used to be freq_cis_imag (for RoPE)

    w.wcls = sharedWeights ? w.token_embedding_table : Array(ptr)
}


func read_checkpoint(checkpoint: String, config: inout Config, weights: inout TransformerWeights, fd: inout Int32, data: inout UnsafeMutablePointer<Float>?, file_size: inout Int) {
    guard let file = fopen(checkpoint, "rb") else {
        print("Couldn't open file \(checkpoint)")
        exit(EXIT_FAILURE)
    }
    
    // read in the config header
    if fread(&config, MemoryLayout<Config>.size, 1, file) != 1 {
        exit(EXIT_FAILURE)
    }
    
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = config.vocab_size > 0 ? 1 : 0
    config.vocab_size = abs(config.vocab_size)
    
    // figure out the file size
    fseek(file, 0, SEEK_END) // move file pointer to end of file
    file_size = ftell(file) // get the file size, in bytes
    fclose(file)
    
    // memory map the Transformer weights into the data pointer
    fd = open(checkpoint, O_RDONLY) // open in read only mode
    if fd == -1 {
        print("open failed!")
        exit(EXIT_FAILURE)
    }
    var ptr = mmap(nil, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    guard ptr != nil else {
        print("mmap failed!")
        exit(EXIT_FAILURE)
    }
    data = ptr!.assumingMemoryBound(to: Float.self)
    let weights_ptr = data!.advanced(by: MemoryLayout<Config>.size / MemoryLayout<Float>.size)
    var weights_array = Array(UnsafeBufferPointer(start: weights_ptr, count: file_size / MemoryLayout<Float>.size))
    memoryMapWeights(w: &weights, p: config, ptr: &weights_array, sharedWeights: shared_weights == 1)
}

print("Hello, World!")

