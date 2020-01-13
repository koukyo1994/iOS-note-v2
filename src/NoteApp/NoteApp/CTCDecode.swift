//
//  CTCDecode.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 12.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation


let CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+-./:<=>?@[\\]^_{}~'"


func decode(_ prediction: [Int], max: Int) -> String {
    var decodedArray = [Character]()
    var previous: Int = 0
    let characterIndex = CHARS.startIndex
    for p in prediction {
        if p == max {
            previous = p
            continue
        }
        if p == previous {
            continue
        }
        let index = CHARS.index(characterIndex, offsetBy: p - 1)
        decodedArray.append(CHARS[index])
        previous = p
    }
    
    return String(decodedArray)
}
