//
//  ViewController.swift
//  NoteApp
//
//  Created by 荒居秀尚 on 09.01.20.
//  Copyright © 2020 荒居秀尚. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    private let navBarCoefficient: CGFloat = 0.02
    
    let canvasView = CanvasView()
    let ruledLineView = UIImageView()

    override func viewDidLoad() {
        super.viewDidLoad()
        // read json
        do {
            if let jsonData = try readJson() {
                canvasView.inverseIndex = try jsonToDict(jsonData: jsonData)
            }
        } catch {
            fatalError("Can't load inverse index.")
        }
        
        // read word
        do {
            if let wordList = try readTxt() {
                canvasView.wordList = wordList
            }
        } catch {
            fatalError("Can't load word list.")
        }
        
        // background
        ruledLineView.backgroundColor = .white
        ruledLineView.frame = getCanvasFrame()
        ruledLineView.isUserInteractionEnabled = false
        ruledLineView.clipsToBounds = true
        ruledLineView.drawRuledLine()
        view.addSubview(ruledLineView)
        
        // canvas
        canvasView.backgroundColor = .clear
        canvasView.frame = getCanvasFrame()
        canvasView.isUserInteractionEnabled = true
        canvasView.clipsToBounds = true
        view.addSubview(canvasView)
        setNavigationBar()
    }
    
    func setNavigationBar() {
        let screenSize: CGRect = UIScreen.main.bounds
        let navBar = UINavigationBar(frame: CGRect(x: 0, y: 0, width: screenSize.width, height: screenSize.height * navBarCoefficient))
        let navItem = UINavigationItem(title: "Navigation Area")
        let barButton = UIBarButtonItem(title: "clear", style: .plain, target: self, action: #selector(self.clear))
        navItem.rightBarButtonItem = barButton
        navBar.setItems([navItem], animated: false)
        view.addSubview(navBar)
    }

    func getCanvasFrame() -> CGRect {
        let screenSize = UIScreen.main.bounds
        return CGRect(x: 0, y: screenSize.height * navBarCoefficient, width: screenSize.width, height: screenSize.height * (1 - navBarCoefficient))
    }
    
    @objc private func clear(_ sender: UIBarButtonItem) {
        canvasView.clear()
    }
}

