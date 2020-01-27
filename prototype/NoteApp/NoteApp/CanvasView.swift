//
//  CanvasView.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 11.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import Foundation
import UIKit
import Vision
import CoreML


class CanvasView: UIImageView {
    private let baseSize: CGFloat = 3.0
    private var color: UIColor = .black
    
    private var ruledLineHeight: CGFloat = 30.0

    private var touchDate = Date()
    private var lastLocation = CGPoint()
    private var touchEvents = [(interval: Double, distance: Double, point: CGPoint)]()
    
    private let roiThreshold = 1.0
    private let distanceThreshold = 30.0
    private var currentROI = CGRect()
    
    public var inverseIndex: [String: [Int]]!
    public var wordList: [String]!
    
    public var detectWithCoreML = true
    private var cumulativeTime = 0.0
    
    public var allowFingerDrawing = true

    private func touchType() -> UITouch.TouchType {
        var touchType = UITouch.TouchType.stylus
        if allowFingerDrawing {
            touchType = .direct
        }
        return touchType
    }
    
    // MARK: UITouch Overriding
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        let touch = touches.first!

        if touch.type == touchType() {
            let point = touch.location(in: self)
            
            let date = Date()
            let interval = touchDate.distance(to: date)
            
            let distanceFromLastLocation = distance(point, lastLocation)
            
            touchDate = date
            lastLocation = point
            touchEvents.append((interval: Double(interval), distance: distanceFromLastLocation, point: point))
        }
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        let touch = touches.first!
        if touch.type == touchType() {
            let point = touch.location(in: self)
            drawLine(touch: touch)
            
            let date = Date()
            let interval = touchDate.distance(to: date)
            
            let distanceFromLastLocation = distance(point, lastLocation)
            
            touchDate = date
            lastLocation = point
            touchEvents.append((interval: Double(interval), distance: distanceFromLastLocation, point: point))
        }
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        DispatchQueue.main.async(execute: {
            if let rect = self.getBoundingBox(
                self.collectRegionOfInterest()) {
                self.displayRegionOfInterest(rect: rect)
                self.currentROI = rect
                if let croppedImage = self.image?.cropRect(rect: rect)?.resize(height: 32)?.padding(in: CGRect(x: 0, y: 0, width: 200, height: 32)) {

                    let startTime = Date()
                    if self.detectWithCoreML {
                        let ciImage = CIImage(image: croppedImage)
                        let handler = VNImageRequestHandler(ciImage: ciImage!)
                        do {
                            try handler.perform([self.recognitionRequest])
                        } catch {
                            print(error)
                        }
                    } else {
                        guard let cgImage = croppedImage.cgImage else {
                            fatalError("can't create CGImage from UIImage")
                        }
                        self.recognitionWithAPI(cgImage)
                    }
                    self.cumulativeTime += startTime.distance(to: Date())
                    print("elapsed \(self.cumulativeTime)")
                }
            }
        })
    }
    
    // MARK: Drawing lines
    private func drawLine(touch: UITouch) {
        UIGraphicsBeginImageContextWithOptions(bounds.size, false, 0.0)
        guard let context = UIGraphicsGetCurrentContext() else {
            return
        }
        
        image?.draw(at: .zero)
        updateContext(context: context, touch: touch)
        image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
    }
    
    private func updateContext(context: CGContext, touch: UITouch) {
        let previousLocation = touch.previousLocation(in: self)
        let location = touch.location(in: self)
        let width = getLineWidth(touch: touch)
        
        context.setLineWidth(width)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.move(to: previousLocation)
        context.addLine(to: location)
        context.strokePath()
    }
    
    private func getLineWidth(touch: UITouch) -> CGFloat {
        var width = baseSize
        if touch.force > 0 {
            width = max(width * (touch.force * 0.6 + 0.2), width)
        }
        
        return width
    }
    
    // MARK: Other public functionality
    func clear() {
        image = nil
        layer.sublayers = nil
        setNeedsDisplay()
        layoutIfNeeded()
        cumulativeTime = 0.0
        touchEvents = []
    }
    
    // MARK: Region of Interest inference
    private func collectRegionOfInterest() -> [CGPoint] {
        var regionOfInterest = [CGPoint]()
        for (interval, dist, point) in touchEvents.reversed() {
            if interval > roiThreshold {
                regionOfInterest.append(point)
                if dist > distanceThreshold {
                    break
                }
            }
            regionOfInterest.append(point)
        }
        
        return regionOfInterest
    }
    
    private func getBoundingBox(_ pointSequence: [CGPoint]) -> CGRect? {
        if pointSequence.count == 0 {
            return nil
        }
        
        var xSequence = [CGFloat]()
        var ySequence = [CGFloat]()
        _ = pointSequence.map { point in
            xSequence.append(point.x)
            ySequence.append(point.y)
        }
        
        let xmax = xSequence.max()!
        let xmin = xSequence.min()!
        let ymax = ySequence.max()!
        let ymin = ySequence.min()!
        
        let width = xmax - xmin
        let height = ymax - ymin
        return CGRect(
            x: xmin - 10,
            y: ymin - 10,
            width: width + 20,
            height: height + 20)
    }
    
    private func displayRegionOfInterest(rect: CGRect) {
        layer.sublayers = nil
        setNeedsDisplay()
        
        let roiLayer = CAShapeLayer()
        roiLayer.frame = rect
        
        // yellow
        roiLayer.fillColor = CGColor(srgbRed: 255, green: 217, blue: 0, alpha: 0.3)
        roiLayer.strokeColor = CGColor(srgbRed: 255, green: 217, blue: 0, alpha: 0.3)
        roiLayer.path = UIBezierPath(rect: CGRect(x: 0, y: 0, width: rect.size.width, height: rect.size.height)).cgPath
        
        layer.addSublayer(roiLayer)
    }
    
    // MARK: Text Recognition
    lazy var recognitionRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: CTCModel().model)
            return VNCoreMLRequest(model: model, completionHandler: self.handleRecognition)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    lazy var recognitionRequestWithAPI: VNRecognizeTextRequest = {
        let request = VNRecognizeTextRequest(completionHandler: self.handleRecognitionUsingAPI)
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en_US"]
        request.usesLanguageCorrection = false
        return request
    }()
    
    private func recognitionWithAPI(_ image: CGImage) {
        let requests = [recognitionRequestWithAPI]
        let imageRequestHandler = VNImageRequestHandler(cgImage: image, options: [:])
        try! imageRequestHandler.perform(requests)
    }
    
    private func handleRecognition(request: VNRequest, error: Error?) {
        guard let observations = request.results
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        let features = observations as! [VNCoreMLFeatureValueObservation]

        do {
            let multiArray = try features[0].featureValue.multiArrayValue!.reshaped(to: [50, 91])
            var argMaxArray = [Int]()

            for i in 0..<50 {
                let sliced = multiArray.slice([.select(i), .slice])
                let pointer = UnsafeMutablePointer<Double>(OpaquePointer(sliced.dataPointer))
                let (argMax, _) = argmax(pointer, count: 91)
                argMaxArray.append(argMax)
            }
            let subArray = argMaxArray.prefix(upTo: 30)
            let decoded = decode(Array(subArray), max: 90)
            print(decoded)
            let candidate = getWordCandidate(observation: decoded)
            if candidate.count > 0 {
                if candidate.count > 10 {
                    print(candidate[..<10])
                } else {
                    print(candidate)
                }
            }
            displayAutoComplete(candidateWords: candidate)
        } catch {
            print(error)
        }
    }
    
    private func handleRecognitionUsingAPI(request: VNRequest?, error: Error?) {
        guard let observations = request?.results as? [VNRecognizedTextObservation] else {
            return
        }
        if observations.count > 0 {
            if let inferedText = observations[0].topCandidates(1).first?.string {
                print(inferedText)
                let candidate = getWordCandidate(observation: inferedText)
                if candidate.count > 0 {
                    if candidate.count > 10 {
                        print(candidate[..<10])
                    } else {
                        print(candidate)
                    }
                }
                displayAutoComplete(candidateWords: candidate)
            }
        }
    }
    
    // MARK: AutoComplete
    private func getWordCandidate(observation: String) -> [String] {
        var candidateWords = [String]()
        if let candidateIndex = inverseIndex[observation] {
            _ = candidateIndex.map {
                candidateWords.append(wordList[$0])
            }
        }
        return candidateWords
    }
    
    private func displayAutoComplete(candidateWords: [String]) {
        if candidateWords.count > 0 {
            let charCount = candidateWords[0].count
            let candidateDisplayLayer = CATextLayer()
            let rect = CGRect(x: self.currentROI.minX, y: self.currentROI.minY-30, width: CGFloat(charCount * 30), height: 30)
            candidateDisplayLayer.frame = rect
            candidateDisplayLayer.font = UIFont(name: "Apple SD Gothic Neo", size: 7)
            candidateDisplayLayer.foregroundColor = UIColor.gray.cgColor
            candidateDisplayLayer.string = candidateWords[0]
            
            layer.addSublayer(candidateDisplayLayer)
        }
    }
}
