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
        guard let context = UIGraphicsGetCurrentContext() else {
            return nil
        }
        
        // white
        context.setFillColor(CGColor(srgbRed: 255, green: 255, blue: 255, alpha: 1.0))
        context.fill(inRect)
        draw(at: .zero)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
    
    func resize(to: CGSize) -> UIImage? {
        let widthRatio = to.width / size.width
        let heightRatio = to.height / size.height
        let ratio = widthRatio < heightRatio ? widthRatio : heightRatio
        
        let resizedSize = CGSize(width: size.width * ratio, height: size.height * ratio)
        UIGraphicsBeginImageContextWithOptions(resizedSize, false, 0.0)
        draw(in: CGRect(origin: .zero, size: resizedSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func resize(height: CGFloat) -> UIImage? {
        let heightRatio = height / size.height
        let resizedSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        UIGraphicsBeginImageContextWithOptions(resizedSize, false, 0.0)
        draw(in: CGRect(origin: .zero, size: resizedSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func resize(width: CGFloat) -> UIImage? {
        let widthRatio = width / size.width
        let resizedSize = CGSize(width: size.width * widthRatio, height: size.height * widthRatio)
        UIGraphicsBeginImageContextWithOptions(resizedSize, false, 0.0)
        draw(in: CGRect(origin: .zero, size: resizedSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
}
