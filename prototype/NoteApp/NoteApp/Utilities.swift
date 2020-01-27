//
//  Utilities.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 11.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation
import UIKit


func distance(_ point1: CGPoint, _ point2: CGPoint) -> Double {
    let xdist = Double(point1.x - point2.x)
    let ydist = Double(point1.y - point2.y)
    
    return sqrt(xdist * xdist + ydist * ydist)
}


