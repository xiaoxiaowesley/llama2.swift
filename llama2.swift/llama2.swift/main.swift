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
    var dim: Int  // transformer dimension
    var hidden_dim: Int  // for ffn layers
    var n_layers: Int  // number of layers
    var n_heads: Int  // number of query heads
    var n_kv_heads: Int  // number of key/value heads (can be < query heads because of multiquery)
    var vocab_size: Int  // vocabulary size, usually 256 (byte-level)
    var seq_len: Int  // max sequence length

    init() {
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab_size = 0
        self.seq_len = 0
    }
}

struct TransformerWeights {
    // token embedding table
    var token_embedding_table: [Float]  // (vocab_size, dim)
    // weights for rmsnorms
    var rms_att_weight: [Float]  // (layer, dim) rmsnorm weights
    var rms_ffn_weight: [Float]  // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    var wq: [Float]  // (layer, dim, n_heads * head_size)
    var wk: [Float]  // (layer, dim, n_kv_heads * head_size)
    var wv: [Float]  // (layer, dim, n_kv_heads * head_size)
    var wo: [Float]  // (layer, n_heads * head_size, dim)
    // weights for ffn
    var w1: [Float]  // (layer, hidden_dim, dim)
    var w2: [Float]  // (layer, dim, hidden_dim)
    var w3: [Float]  // (layer, hidden_dim, dim)
    // final rmsnorm
    var rms_final_weight: [Float]  // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    var wcls: [Float]?

    init() {
        self.token_embedding_table = []
        self.rms_att_weight = []
        self.rms_ffn_weight = []
        self.wq = []
        self.wk = []
        self.wv = []
        self.wo = []
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.rms_final_weight = []
        self.wcls = []

    }
}

struct RunState {
    // current wave of activations
    var x: [Float]  // activation at current time stamp (dim,)
    var xb: [Float]  // same, but inside a residual branch (dim,)
    var xb2: [Float]  // an additional buffer just for convenience (dim,)
    var hb: [Float]  // buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: [Float]  // buffer for hidden dimension in the ffn (hidden_dim,)
    var q: [Float]  // query (dim,)
    var k: [Float]  // key (dim,)
    var v: [Float]  // value (dim,)
    var att: [Float]  // buffer for scores/attention values (n_heads, seq_len)
    var logits: [Float]  // output logits
    // kv cache
    var key_cache: [Float]  // (layer, seq_len, dim)
    var value_cache: [Float]  // (layer, seq_len, dim)

    //constructor
    init() {
        self.x = []
        self.xb = []
        self.xb2 = []
        self.hb = []
        self.hb2 = []
        self.q = []
        self.k = []
        self.v = []
        self.att = []
        self.logits = []
        self.key_cache = []
        self.value_cache = []
    }
}

struct Transformer {
    var config: Config  // the hyperparameters of the architecture (the blueprint)
    var weights: TransformerWeights  // the weights of the model
    var state: RunState  // buffers for the "wave" of activations in the forward pass
    var fd: Int32  // file descriptor for memory mapping
    var data: UnsafeMutablePointer<Float>?  // memory mapped data pointer
    var fileSize: Int  // size of the checkpoint file in bytes

    //constructor
    init() {
        self.config = Config()
        self.weights = TransformerWeights()
        self.state = RunState()
        self.fd = 0
        self.data = nil
        self.fileSize = 0
    }
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
        || s.key_cache.isEmpty || s.value_cache.isEmpty || s.att.isEmpty || s.logits.isEmpty
    {
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

func memoryMapWeights(
    w: inout TransformerWeights, p: Config, ptr: inout [Float], sharedWeights: Bool
) {
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

    ptr.removeFirst(seqLenHeadSize)  // skip what used to be freq_cis_real (for RoPE)
    ptr.removeFirst(seqLenHeadSize)  // skip what used to be freq_cis_imag (for RoPE)

    w.wcls = sharedWeights ? w.token_embedding_table : Array(ptr)
}

func read_checkpoint(
    checkpoint: String, config: inout Config, weights: inout TransformerWeights, fd: inout Int32,
    data: inout UnsafeMutablePointer<Float>?, file_size: inout Int
) {
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
    fseek(file, 0, SEEK_END)  // move file pointer to end of file
    file_size = ftell(file)  // get the file size, in bytes
    fclose(file)

    // memory map the Transformer weights into the data pointer
    fd = open(checkpoint, O_RDONLY)  // open in read only mode
    if fd == -1 {
        print("open failed!")
        exit(EXIT_FAILURE)
    }
    let ptr = mmap(nil, file_size, PROT_READ, MAP_PRIVATE, fd, 0)
    guard ptr != nil else {
        print("mmap failed!")
        exit(EXIT_FAILURE)
    }
    data = ptr!.assumingMemoryBound(to: Float.self)
    let weights_ptr = data!.advanced(by: MemoryLayout<Config>.size / MemoryLayout<Float>.size)
    var weights_array = Array(
        UnsafeBufferPointer(start: weights_ptr, count: file_size / MemoryLayout<Float>.size))
    memoryMapWeights(
        w: &weights, p: config, ptr: &weights_array, sharedWeights: shared_weights == 1)
}

func buildTransformer(_ checkpointPath: String) -> Transformer {
    var t: Transformer = Transformer()

    // read in the Config and the Weights from the checkpoint
    read_checkpoint(
        checkpoint: checkpointPath, config: &t.config, weights: &t.weights, fd: &t.fd,
        data: &t.data, file_size: &t.fileSize)
    // allocate the RunState buffers
    mallocRunState(s: &t.state, p: t.config)

    return t
}

func freeTransformer(t: inout Transformer) {
    // close the memory mapping
    if t.data != nil {
        munmap(t.data, t.fileSize)
        t.data = nil
    }
    if t.fd != -1 {
        close(t.fd)
        t.fd = -1
    }
    // free the RunState buffers
    freeRunState(s: &t.state)
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

func rmsnorm(o: inout [Float], x: [Float], weight: [Float], size: Int) {
    // calculate sum of squares
    var ss: Float = 0.0
    for j in 0..<size {
        ss += x[j] * x[j]
    }
    ss /= Float(size)
    ss += 1e-5
    ss = 1.0 / sqrt(ss)
    // normalize and scale
    for j in 0..<size {
        o[j] = weight[j] * (ss * x[j])
    }
}

func softmax(_ x: inout [Float]) {
    // find max value (for numerical stability)
    let maxVal = x.max() ?? 0.0

    // exp and sum
    var sum: Float = 0.0
    for i in 0..<x.count {
        x[i] = exp(x[i] - maxVal)
        sum += x[i]
    }

    // normalize
    for i in 0..<x.count {
        x[i] /= sum
    }
}

func matmul(_ xout: inout [Float], _ x: [Float], _ w: [Float], _ n: Int, _ d: Int) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    DispatchQueue.concurrentPerform(iterations: d) { i in
        var val: Float = 0.0
        for j in 0..<n {
            val += w[i * n + j] * x[j]
        }
        xout[i] = val
    }
}

func forward(transformer: inout Transformer, token: Int, pos: Int) -> [Float] {
    // a few convenience variables
    let p = transformer.config
    let w = transformer.weights
    var s = transformer.state
    var x = s.x
    let dim = p.dim
    let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads
    let kv_mul = p.n_heads / p.n_kv_heads  // integer multiplier of the kv sharing in multiquery
    let hidden_dim = p.hidden_dim
    let head_size = dim / p.n_heads

    // copy the token embedding into x
    let content_row = Array(w.token_embedding_table[(token * dim)..<(token * dim + dim)])
    x = content_row

    // forward all the layers
    for l in 0..<p.n_layers {
        // attention rmsnorm
        rmsnorm(
            o: &s.xb, x: x, weight: Array(w.rms_att_weight[(l * dim)..<(l * dim + dim)]), size: dim)

        // key and value point to the kv cache
        let loff = l * p.seq_len * kv_dim  // kv cache layer offset for convenience
        s.k = Array(s.key_cache[(loff + pos * kv_dim)..<(loff + pos * kv_dim + kv_dim)])
        s.v = Array(s.value_cache[(loff + pos * kv_dim)..<(loff + pos * kv_dim + kv_dim)])

        // qkv matmuls for this position
        matmul(&s.q, s.xb, Array(w.wq[(l * dim * dim)..<(l * dim * dim + dim * dim)]), dim, dim)
        matmul(
            &s.k, s.xb, Array(w.wk[(l * dim * kv_dim)..<(l * dim * kv_dim + dim * kv_dim)]), dim,
            kv_dim)
        matmul(
            &s.v, s.xb, Array(w.wv[(l * dim * kv_dim)..<(l * dim * kv_dim + dim * kv_dim)]), dim,
            kv_dim)

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in stride(from: 0, to: dim, by: 2) {
            let head_dim = i % head_size
            let freq = 1.0 / pow(10000.0, Float(head_dim) / Float(head_size))
            let val = Float(pos) * freq
            let fcr = cos(val)
            let fci = sin(val)
            let rotn = i < kv_dim ? 2 : 1  // how many vectors? 2 = q & k, 1 = q only
            for v in 0..<rotn {
                var vec = v == 0 ? s.q : s.k  // the vector to rotate (query or key)
                let v0 = vec[i]
                let v1 = vec[i + 1]
                vec[i] = v0 * fcr - v1 * fci
                vec[i + 1] = v0 * fci + v1 * fcr
            }
        }

        // multihead attention. iterate over all heads
        for h in 0..<p.n_heads {
            // get the query vector for this head
            let q = Array(s.q[(h * head_size)..<(h * head_size + head_size)])
            // attention scores for this head
            var att = Array(s.att[(h * p.seq_len)..<(h * p.seq_len + p.seq_len)])
            // iterate over all timesteps, including the current one
            for t in 0...pos {
                // get the key vector for this head and at this timestep
                let k = Array(s.key_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..<(loff + t * kv_dim + (h / kv_mul) * head_size + head_size)])
                // calculate the attention score as the dot product of q and k
                var score: Float = 0.0
                for i in 0..<head_size {
                    score += q[i] * k[i]
                }
                score /= sqrt(Float(head_size))
                // save the score to the attention buffer
                att[t] = score
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&att)

            // weighted sum of the values, store back into xb
            var xb = Array(s.xb[(h * head_size)..<(h * head_size + head_size)])
            for t in 0...pos {
                // get the value vector for this head and at this timestep
                let v = Array(s.value_cache[(loff + t * kv_dim + (h / kv_mul) * head_size)..<(loff + t * kv_dim + (h / kv_mul) * head_size + head_size)])
                // get the attention weight for this timestep
                let a = att[t]
                // accumulate the weighted value into xb
                for i in 0..<head_size {
                    xb[i] += a * v[i]
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(&s.xb2, s.xb, Array(w.wo[(l * dim * dim)..<(l * dim * dim + dim * dim)]), dim, dim)

        // residual connection back into x
        for i in 0..<dim {
            x[i] += s.xb2[i]
        }

        // ffn rmsnorm
        rmsnorm(
            o: &s.xb, x: x, weight: Array(w.rms_ffn_weight[(l * dim)..<(l * dim + dim)]), size: dim)

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            &s.hb, s.xb,
            Array(w.w1[(l * dim * hidden_dim)..<(l * dim * hidden_dim + dim * hidden_dim)]), dim,
            hidden_dim)
        matmul(
            &s.hb2, s.xb,
            Array(w.w3[(l * dim * hidden_dim)..<(l * dim * hidden_dim + dim * hidden_dim)]), dim,
            hidden_dim)

        // SwiGLU non-linearity
        for i in 0..<hidden_dim {
            var val = s.hb[i]
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0 / (1.0 + exp(-val)))
            // elementwise multiply with w3(x)
            val *= s.hb2[i]
            s.hb[i] = val
        }

        // final matmul to get the output of the ffn
        matmul(
            &s.xb, s.hb,
            Array(w.w2[(l * dim * hidden_dim)..<(l * dim * hidden_dim + hidden_dim * dim)]),
            hidden_dim, dim)

        // residual connection
        for i in 0..<dim {
            x[i] += s.xb[i]
        }
    }

    // final rmsnorm
    rmsnorm(o: &x, x: x, weight: w.rms_final_weight, size: dim)

    // classifier into logits  //TODO:待确认
    matmul(&s.logits, x, w.wcls ?? [], p.dim, p.vocab_size)
    return s.logits
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct TokenIndex {
    var str: String
    var id: Int
    init() {
        self.str = ""
        self.id = 0
    }

    init(str: String, id: Int) {
        self.str = str
        self.id = id
    }
}

struct Tokenizer {
    var vocab: [String]
    var vocabScores: [Float]
    var sortedVocab: [TokenIndex]
    var vocabSize: Int
    var maxTokenLength: UInt
    var bytePieces: [UInt8]  // stores all single-byte strings

    init() {
        self.vocab = []
        self.vocabScores = []
        self.sortedVocab = []
        self.vocabSize = 0
        self.maxTokenLength = 0
        self.bytePieces = Array(repeating: 0, count: 256 * 2)
    }
}

func compare_tokens(_ a: TokenIndex, _ b: TokenIndex) -> Bool {
    return a.str < b.str
}

func buildTokenizer(tokenizerPath: String, vocabSize: Int) -> Tokenizer {

    var t = Tokenizer()
    t.vocabSize = vocabSize
    t.vocab = [String](repeating: "", count: vocabSize)
    t.vocabScores = [Float](repeating: 0.0, count: vocabSize)
    t.sortedVocab = []

    for i in 0..<256 {
        t.bytePieces[i * 2] = UInt8(i)
        t.bytePieces[i * 2 + 1] = 0
    }

    do {
        let fileData = try Data(
            contentsOf: URL(fileURLWithPath: tokenizerPath), options: .mappedIfSafe)
        var readPosition = 0

        t.maxTokenLength = fileData.withUnsafeBytes {
            $0.load(fromByteOffset: readPosition, as: UInt.self)
        }
        readPosition += MemoryLayout<UInt>.size

        for i in 0..<vocabSize {
            t.vocabScores[i] = fileData.withUnsafeBytes {
                $0.load(fromByteOffset: readPosition, as: Float.self)
            }
            readPosition += MemoryLayout<Float>.size

            let len = fileData.withUnsafeBytes {
                $0.load(fromByteOffset: readPosition, as: Int.self)
            }
            readPosition += MemoryLayout<Int>.size

            let tokenData = fileData.subdata(in: readPosition..<(readPosition + len))
            t.vocab[i] = String(data: tokenData, encoding: .utf8) ?? ""
            readPosition += len
        }
    } catch {
        print("Couldn't load \(tokenizerPath)")
        exit(EXIT_FAILURE)
    }

    return t

}

func freeTokenizer(t: inout Tokenizer) {
    t.vocab.removeAll()
    t.vocabScores.removeAll()
    t.sortedVocab.removeAll()
}
func decode(t: inout Tokenizer, prevToken: Int, token: Int) -> String {
    var piece = t.vocab[token]
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if prevToken == 1 && piece.first == " " { piece.removeFirst() }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    if let range = piece.range(of: "<0x[0-9A-Fa-f]{2}>", options: .regularExpression),
        let byteVal = UInt8(piece[range].dropFirst(3).dropLast(1), radix: 16)
    {
        piece = String(bytes: [t.bytePieces[Int(byteVal) * 2]], encoding: .utf8) ?? ""
    }
    return piece
}

func strLookup(str: String, sortedVocab: inout [TokenIndex], vocabSize: Int) -> Int {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    let tok = TokenIndex(str: str, id: 0)  // acts as the key to search for

    let res = sortedVocab.firstIndex(where: { $0.str == tok.str })

    return res != nil ? sortedVocab[res!].id : -1
}

func encode(
    t: inout Tokenizer, text: String, bos: Int8, eos: Int8, tokens: inout [Int], n_tokens: inout Int
) {
    if text.isEmpty {
        print("cannot encode empty text")
        exit(EXIT_FAILURE)
    }

    if t.sortedVocab.isEmpty {
        t.sortedVocab = Array(repeating: TokenIndex(str: "", id: 0), count: t.vocabSize)
        for i in 0..<t.vocabSize {
            t.sortedVocab[i].str = t.vocab[i]
            t.sortedVocab[i].id = i
        }
        t.sortedVocab.sort(by: compare_tokens)
    }

    var strBuffer = Array(
        repeating: Character(UnicodeScalar(0)!), count: Int(t.maxTokenLength * 2 + 1 + 2))
    var strLen = 0

    n_tokens = 0

    if bos != 0 {
        tokens[n_tokens] = 1
        n_tokens += 1
    }

    if !text.isEmpty {
        let dummyPrefix = strLookup(str: " ", sortedVocab: &t.sortedVocab, vocabSize: t.vocabSize)
        tokens[n_tokens] = dummyPrefix
        n_tokens += 1
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point    Last code point    Byte 1    Byte 2    Byte 3    Byte 4
    // U+0000    U+007F        0xxxxxxx
    // U+0080    U+07FF        110xxxxx    10xxxxxx
    // U+0800    U+FFFF        1110xxxx    10xxxxxx    10xxxxxx
    // U+10000    U+10FFFF    11110xxx    10xxxxxx    10xxxxxx    10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for c in text.utf8 {
        if c & 0xC0 != 0x80 {
            strLen = 0
        }

        strBuffer[strLen] = Character(UnicodeScalar(c))
        strLen += 1
        strBuffer[strLen] = Character(UnicodeScalar(0))

        if (c + 1 & 0xC0) == 0x80 && strLen < 4 {
            continue
        }

        let id = strLookup(
            str: String(strBuffer[0..<strLen]), sortedVocab: &t.sortedVocab, vocabSize: t.vocabSize)

        if id != -1 {
            tokens[n_tokens] = id
            n_tokens += 1
        } else {
            for i in 0..<strLen {
                tokens[n_tokens] = Int(strBuffer[i].asciiValue!) + 3
                n_tokens += 1
            }
        }
        strLen = 0
    }

    while true {
        var bestScore: Float = -1e10
        var bestId = -1
        var bestIdx = -1

        for i in 0..<(n_tokens - 1) {
            let strBuffer = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]]
            let id = strLookup(str: strBuffer, sortedVocab: &t.sortedVocab, vocabSize: t.vocabSize)
            if id != -1 && t.vocabScores[id] > bestScore {
                bestScore = t.vocabScores[id]
                bestId = id
                bestIdx = i
            }
        }

        if bestIdx == -1 {
            break
        }

        tokens[bestIdx] = bestId
        for i in (bestIdx + 1)..<(n_tokens - 1) {
            tokens[i] = tokens[i + 1]
        }
        n_tokens -= 1
    }

    if eos != 0 {
        tokens[n_tokens] = 2
        n_tokens += 1
    }
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

struct ProbIndex {
    var prob: Float
    var index: Int
    init() {
        self.prob = 0.0
        self.index = 0
    }
    init(prob: Float, index: Int) {
        self.prob = prob
        self.index = index
    }
}  // struct used when sorting probabilities during top-p sampling

struct Sampler {
    var vocab_size: Int
    var probindex: [ProbIndex]  // buffer used in top-p sampling
    var temperature: Float
    var topp: Float
    var rng_state: UInt64

    init() {
        self.vocab_size = 0
        self.probindex = []
        self.temperature = 0.0
        self.topp = 0.0
        self.rng_state = 0
    }
}

func sampleArgmax(probabilities: [Float]) -> Int {
    // return the index that has the highest probability
    var maxIndex = 0
    var maxProbability = probabilities[0]
    for i in 1..<probabilities.count {
        if probabilities[i] > maxProbability {
            maxIndex = i
            maxProbability = probabilities[i]
        }
    }
    return maxIndex
}

func sampleMult(probabilities: [Float], n: Int, coin: Float) -> Int {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    var cdf: Float = 0.0
    for i in 0..<n {
        cdf += probabilities[i]
        if coin < cdf {
            return i
        }
    }
    return n - 1  // in case of rounding errors
}

func compare(a: UnsafeRawPointer, b: UnsafeRawPointer) -> Int {
    let a_ = a.assumingMemoryBound(to: ProbIndex.self)
    let b_ = b.assumingMemoryBound(to: ProbIndex.self)
    if a_.pointee.prob > b_.pointee.prob { return -1 }
    if a_.pointee.prob < b_.pointee.prob { return 1 }
    return 0
}

func sample_topp(
    probabilities: inout [Float], n: Int, topp: Float, probindex: inout [ProbIndex], coin: Float
) -> Int {
    var n0 = 0
    let cutoff = (1.0 - topp) / Float(n - 1)
    for i in 0..<n {
        if probabilities[i] >= cutoff {
            probindex[n0].index = i
            probindex[n0].prob = probabilities[i]
            n0 += 1
        }
    }
    probindex[0..<n0].sort { $0.prob > $1.prob }

    var cumulative_prob: Float = 0.0
    var last_idx = n0 - 1
    for i in 0..<n0 {
        cumulative_prob += probindex[i].prob
        if cumulative_prob > topp {
            last_idx = i
            break
        }
    }

    let r = coin * cumulative_prob
    var cdf: Float = 0.0
    for i in 0...last_idx {
        cdf += probindex[i].prob
        if r < cdf {
            return probindex[i].index
        }
    }
    return probindex[last_idx].index
}

func buildSampler(vocabSize: Int, temperature: Float, topp: Float, rngSeed: UInt64) -> Sampler {
    var sampler = Sampler()
    sampler.vocab_size = vocabSize
    sampler.temperature = temperature
    sampler.topp = topp
    sampler.rng_state = rngSeed
    sampler.probindex = Array(repeating: ProbIndex(prob: 0.0, index: 0), count: sampler.vocab_size)
    return sampler
}

func freeSampler(sampler: inout Sampler) {
    sampler.probindex = []
}

func randomU32(state: inout UInt64) -> UInt32 {
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    return UInt32((state * 0x2545_F491_4F6C_DD1D) >> 32)
}

func randomF32(state: inout UInt64) -> Float {
    return Float(randomU32(state: &state) >> 8) / 16777216.0
}

func sample(sampler: inout Sampler, logits: inout [Float]) -> Int {
    var next: Int
    if sampler.temperature == 0.0 {
        next = sampleArgmax(probabilities: logits)
    } else {
        for q in 0..<sampler.vocab_size {
            logits[q] /= sampler.temperature
        }
        softmax(&logits)
        let coin = randomF32(state: &sampler.rng_state)
        if sampler.topp <= 0 || sampler.topp >= 1 {
            next = sampleMult(probabilities: logits, n: sampler.vocab_size, coin: coin)
        } else {
            next = sample_topp(
                probabilities: &logits, n: sampler.vocab_size, topp: sampler.topp,
                probindex: &sampler.probindex, coin: coin)
        }
    }
    return next
}

// ----------------------------------------------------------------------------
// utilities: time

func timeInMs() -> Int64 {
    let time = Date().timeIntervalSince1970
    return Int64(time * 1000)
}

// ----------------------------------------------------------------------------
// generation loop
func generate(
    transformer: Transformer, tokenizer: Tokenizer, sampler: inout Sampler, prompt: String?,
    steps: Int
) {
    let prompt = prompt ?? ""

    // encode the (string) prompt into tokens sequence
    var numPromptTokens = 0
    var promptTokens = [Int](repeating: 0, count: prompt.count + 3)  // +3 for '\0', ?BOS, ?EOS

    var tokenizer = tokenizer  // Make the 'tokenizer' parameter mutable
    var transformer = transformer  // Make the 'transformer' parameter mutable
    encode(
        t: &tokenizer, text: prompt, bos: 1, eos: 0, tokens: &promptTokens,
        n_tokens: &numPromptTokens)
    if numPromptTokens < 1 {
        print("something is wrong, expected at least 1 prompt token")
        exit(EXIT_FAILURE)
    }

    // start the main loop
    var start: Int64 = 0  // used to time our code, only initialized after first iteration
    var next: Int  // will store the next token in the sequence
    var token = promptTokens[0]  // kick off with the first token in the prompt
    var pos = 0  // position in the sequence
    while pos < steps {

        // forward the transformer to get logits for the next token
        var logits = forward(transformer: &transformer, token: token, pos: pos)

        // advance the state machine
        if pos < numPromptTokens - 1 {
            // if we are still processing the input prompt, force the next prompt token
            next = promptTokens[pos + 1]
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler: &sampler, logits: &logits)
        }
        pos += 1

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 { break }

        // print the token as string, decode it with the Tokenizer object
        let piece = decode(t: &tokenizer, prevToken: token, token: next)
        print(piece)
        fflush(stdout)
        token = next

        // init the timer here because the first iteration can be slower
        if start == 0 { start = timeInMs() }
    }
    print("\n")

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end = timeInMs()
        print("achieved tok/s: \(Double(pos-1) / Double(end-start)*1000)")
    }

    promptTokens = []
}

func readStdin(guide: String) -> String? {
    print(guide, terminator: "")
    let buffer = readLine()
    return buffer
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

func chat(
    transformer: Transformer, tokenizer: Tokenizer, sampler: inout Sampler, cliUserPrompt: String?,
    cliSystemPrompt: String?, steps: Int
) {
    var systemPrompt = ""
    var userPrompt = ""
    var renderedPrompt = ""
    var numPromptTokens = 0
    var promptTokens = [Int](repeating: 0, count: 1152)
    var userIdx: Int = 0

    var userTurn: Bool = true
    var next: Int = 0
    var token: Int = 0
    var pos = 0

    var tokenizer = tokenizer  // Make the 'tokenizer' parameter mutable
    var transformer = transformer  // Make the 'transformer' parameter mutable

    while pos < steps {
        if userTurn {
            if pos == 0 {
                if let cliSystemPrompt = cliSystemPrompt {
                    systemPrompt = cliSystemPrompt
                } else {
                    systemPrompt = readStdin(guide: "Enter system prompt (optional): ") ?? ""
                }
            }
            if pos == 0 && cliUserPrompt != nil {
                userPrompt = cliUserPrompt!
            } else {
                userPrompt = readStdin(guide: "User: ") ?? ""
            }
            if pos == 0 && !systemPrompt.isEmpty {
                let systemTemplate = "[INST] <<SYS>>\n%@\n<</SYS>>\n\n%@ [/INST]"
                renderedPrompt = String(format: systemTemplate, systemPrompt, userPrompt)
            } else {
                let userTemplate = "[INST] %@ [/INST]"
                renderedPrompt = String(format: userTemplate, userPrompt)
            }
            encode(
                t: &tokenizer, text: renderedPrompt, bos: 1, eos: 0, tokens: &promptTokens,
                n_tokens: &numPromptTokens)
            userIdx = 0
            userTurn = false
            print("Assistant: ", terminator: "")
        }

        if userIdx < numPromptTokens {
            token = promptTokens[userIdx]
            userIdx += 1
        } else {
            token = next
        }
        if token == 2 { userTurn = true }

        var logits = forward(transformer: &transformer, token: token, pos: pos)
        next = sample(sampler: &sampler, logits: &logits)
        pos += 1

        if userIdx >= numPromptTokens && next != 2 {
            let piece = decode(t: &tokenizer, prevToken: token, token: next)
            print(piece, terminator: "")
            fflush(stdout)
        }
        if next == 2 { print("\n") }
    }
    print("\n")
    promptTokens = []
}

func errorUsage() {
    print("Usage:   run <checkpoint> [options]")
    print("Example: run model.bin -n 256 -i \"Once upon a time\"")
    print("Options:")
    print("  -t <float>  temperature in [0,inf], default 1.0")
    print("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9")
    print("  -s <int>    random seed, default time(NULL)")
    print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len")
    print("  -i <string> input prompt")
    print("  -z <string> optional path to custom tokenizer")
    print("  -m <string> mode: generate|chat, default: generate")
    print("  -y <string> (optional) system prompt in chat mode")
    exit(EXIT_FAILURE)
}

var checkpointPath: String? = nil
var tokenizerPath: String = "tokenizer.bin"
var temperature: Float = 1.0
var topp: Float = 0.9
var steps: Int = 256
var prompt: String? = nil
var rngSeed: UInt64 = 0
var mode: String = "generate"
var systemPrompt: String? = nil

let args = CommandLine.arguments
if args.count >= 2 { checkpointPath = args[1] } else { errorUsage() }
guard let checkpointPath = checkpointPath else {
    errorUsage()
    exit(EXIT_FAILURE)
}
for i in stride(from: 2, to: args.count, by: 2) {
    if i + 1 >= args.count { errorUsage() }
    if args[i].first != "-" { errorUsage() }
    if args[i].count != 2 { errorUsage() }
    switch args[i] {
    case "-t": temperature = Float(args[i + 1]) ?? 1.0
    case "-p": topp = Float(args[i + 1]) ?? 0.9
    case "-s": rngSeed = UInt64(args[i + 1]) ?? 0
    case "-n": steps = Int(args[i + 1]) ?? 256
    case "-i": prompt = args[i + 1]
    case "-z": tokenizerPath = args[i + 1]
    case "-m": mode = args[i + 1]
    case "-y": systemPrompt = args[i + 1]
    default: errorUsage()
    }
}

if rngSeed <= 0 { rngSeed = UInt64(Date().timeIntervalSince1970) }
if temperature < 0.0 { temperature = 0.0 }
if topp < 0.0 || topp > 1.0 { topp = 0.9 }
if steps < 0 { steps = 0 }

var transformer = buildTransformer(checkpointPath)
if steps == 0 || steps > transformer.config.seq_len { steps = transformer.config.seq_len }

var tokenizer = buildTokenizer(
    tokenizerPath: tokenizerPath, vocabSize: transformer.config.vocab_size)

var sampler = buildSampler(
    vocabSize: transformer.config.vocab_size, temperature: temperature, topp: topp, rngSeed: rngSeed
)

if mode == "generate" {
    generate(
        transformer: transformer, tokenizer: tokenizer, sampler: &sampler, prompt: prompt,
        steps: steps)
} else if mode == "chat" {
    chat(
        transformer: transformer, tokenizer: tokenizer, sampler: &sampler, cliUserPrompt: prompt,
        cliSystemPrompt: systemPrompt, steps: steps)
} else {
    print("unknown mode: \(mode)")
    errorUsage()
}

freeSampler(sampler: &sampler)
freeTokenizer(t: &tokenizer)
freeTransformer(t: &transformer)

