//
//  AutoCompletionUtils.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 13.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation


func readJson() throws -> Data? {
    guard let pathString = Bundle.main.path(forResource: "index", ofType: "json") else {
        return nil
    }
    let url = URL(fileURLWithPath: pathString)
    
    return try Data(contentsOf: url)
}


func jsonToDict(jsonData: Data) throws -> [String: [Int]] {
    return try (JSONSerialization.jsonObject(with: jsonData, options: []) as? [String: [Int]])!
}


func readTxt() throws -> [String]? {
    guard let pathString = Bundle.main.path(forResource: "words", ofType: "txt") else {
        return nil
    }
    let url = URL(fileURLWithPath: pathString)
    
    do {
        let text = try String(contentsOf: url)
        let wordList = text.split(separator: "\n")
        return wordList.map { String($0) }
    } catch {
        throw error
    }
}
