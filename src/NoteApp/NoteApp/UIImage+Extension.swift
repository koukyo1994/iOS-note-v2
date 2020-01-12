//
//  UIImage+Extension.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 09.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation
import UIKit


extension UIImage {
    func cropRect(rect: CGRect) -> UIImage? {
        guard let cgImage = self.cgImage else {
            return nil
        }
        
        guard let croppedImage = cgImage.cropping(
            to: CGRect(
                x: rect.minX * scale,
                y: rect.minY * scale,
                width: rect.width * scale,
                height: rect.height * scale)) else {
                    return nil
        }
        
        return UIImage(cgImage: croppedImage, scale: scale, orientation: imageOrientation)
    }
    
    func padding(in inRect: CGRect) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(inRect.size, false, 0.0)
        draw(at: .zero)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
}
