//
//  Math.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 12.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation
import Accelerate


public func argmax(_ array: [Float], count: Int? = nil) -> (Int, Float) {
    var maxValue: Float = 0
    var maxIndex: vDSP_Length = 0
    vDSP_maxvi(array, 1, &maxValue, &maxIndex, vDSP_Length(count ?? array.count))
    return (Int(maxIndex), maxValue)
}


public func argmax(_ array: [Double], count: Int? = nil) -> (Int, Double) {
    var maxValue: Double = 0
    var maxIndex: vDSP_Length = 0
    vDSP_maxviD(array, 1, &maxValue, &maxIndex, vDSP_Length(count ?? array.count))
    return (Int(maxIndex), maxValue)
}


public func argmax(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
    var maxValue: Float = 0
    var maxIndex: vDSP_Length = 0
    vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
    return (Int(maxIndex), maxValue)
}


public func argmax(_ ptr: UnsafePointer<Double>, count: Int, stride: Int = 1) -> (Int, Double) {
    var maxValue: Double = 0
    var maxIndex: vDSP_Length = 0
    vDSP_maxviD(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
    return (Int(maxIndex), maxValue)
}


public func sum(_ ptr: UnsafePointer<Float>, count: Int) -> Float {
    var sumValue: Float = 0
    vDSP_sve(ptr, vDSP_Stride(1), &sumValue, vDSP_Length(count))
    return sumValue
}


public func sum(_ ptr: UnsafePointer<Double>, count: Int) -> Double {
    var sumValue: Double = 0
    vDSP_sveD(ptr, vDSP_Stride(1), &sumValue, vDSP_Length(count))
    return sumValue
}
