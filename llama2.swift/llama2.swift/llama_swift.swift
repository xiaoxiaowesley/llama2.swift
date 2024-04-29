//
//  llama_swift.swift
//  llama2.swift
//
//  Created by xiaoxiang's m1 mbp on 2024/4/29.
//

import Foundation


func Generate(_ transformer: inout Transformer,_  tokenizer: inout Tokenizer,_ sampler: inout Sampler, _ prompt: String?, _ steps: Int) {
    generate(&transformer, &tokenizer, &sampler, nil, Int32(steps));
}
