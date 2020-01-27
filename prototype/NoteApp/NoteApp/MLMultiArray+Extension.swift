//
//  MLMultiArray+Extension.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 12.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation
import CoreML


extension MLMultiArray {
    enum Indexing: Equatable {
        case select(Int)
        case slice
    }
    
    @nonobjc public func reshaped(to dimensions: [Int]) throws -> MLMultiArray {
        let newCount = dimensions.reduce(1, *)
        precondition(newCount == count, "Cannot reshape \(shape) to \(dimensions)")
        
        var newStrides = [Int](repeating: 0, count: dimensions.count)
        newStrides[dimensions.count - 1] = 1
        for i in stride(from: dimensions.count - 1, to: 0, by: -1) {
            newStrides[i - 1] = newStrides[i] * dimensions[i]
        }
        
        let newShape_ = dimensions.map { NSNumber(value: $0) }
        let newStrides_ = newStrides.map { NSNumber(value: $0) }
        
        return try MLMultiArray(dataPointer: self.dataPointer,
                            shape: newShape_,
                            dataType: self.dataType,
                            strides: newStrides_)
    }
    
    func slice(_ indexing: [Indexing]) -> MLMultiArray {
        assert(indexing.count == self.shape.count)
        assert(indexing.filter { $0 == Indexing.slice }.count == 1)
        
        var selectDims: [Int: Int] = [:]
        for (i, idx) in indexing.enumerated() {
            if case .select(let select) = idx {
                selectDims[i] = select
            }
        }
        
        return .slice(
            self,
            sliceDim: indexing.firstIndex { $0 == Indexing.slice }!,
            selectDims: selectDims)
    }
    
    static func slice(_ o: MLMultiArray, sliceDim: Int, selectDims: [Int: Int]) -> MLMultiArray {
        assert(selectDims.count + 1 == o.shape.count)
        var shape: [NSNumber] = Array(repeating: 1, count: o.shape.count)
        shape[sliceDim] = o.shape[sliceDim]
        
        let arr = try! MLMultiArray(shape: shape, dataType: .double)
        
        let dstPtr = UnsafeMutablePointer<Double>(OpaquePointer(arr.dataPointer))
        for i in 0..<arr.count {
            var index: [Int] = []
            for j in 0..<shape.count {
                if j == sliceDim {
                    index.append(i)
                } else {
                    index.append(selectDims[j]!)
                }
            }
            dstPtr[i] = o[index as [NSNumber]] as! Double
        }
        return arr
    }
}
