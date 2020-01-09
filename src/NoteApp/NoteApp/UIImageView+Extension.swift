//
//  UIImageView+Extension.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 09.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation
import UIKit


extension UIImageView {
    func drawRuledLine() {
        let frameHeight = frame.size.height
        let frameWidth = frame.size.width
        let ruledLineHeight: CGFloat = 30
        let lineWidth: CGFloat = 1.0
        
        var currentHeight: CGFloat = 0.0
        
        UIGraphicsBeginImageContextWithOptions(bounds.size, false, 0.0)
        guard let context = UIGraphicsGetCurrentContext() else {
            return
        }
        
        context.setLineWidth(lineWidth)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.setFillColor(CGColor(srgbRed: 240, green: 240, blue: 240, alpha: 0.4))
        while currentHeight < frameHeight {
            currentHeight += ruledLineHeight
            context.move(to: CGPoint(x: 0, y: currentHeight))
            context.addLine(to: CGPoint(x: frameWidth, y: currentHeight))
            context.strokePath()
        }
        
        image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
    }
}
