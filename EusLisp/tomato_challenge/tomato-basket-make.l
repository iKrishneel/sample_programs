;;
;; Make Tomato basket on the IRT viewer
;;

;; Note
;; To remove objects from the irtviewer type command
;; (objects (remove *box* (send *irtviewer* :objects)))
;; --> The first object is the target set object  and the second one to remove


;; Function to put box on the robot
(defun make-tomato-box ()
  
  ;;Clear previous box
  (if (boundp '*box*)
    (remove-box)
    )
  
  ;; making box  250, 350 80
  (setq box0 (make-cube 250 350 100))
  (setq box1 (make-cube 220 300 90))
  (send box1 :translate (float-vector 0 0 10))
  (setq box2 (body- box0 box1))
  ;;(objects (list box2))
  
  
  ;; reaching hand to the box
  ;(setq *box* (make-cube 300 300 300))
  
  (setq *box* (body- box2))
  (send *box* :locate (float-vector 220 0 475))
  (send *hrp2* :assoc *box*)
  (objects *box*)
  (send (send *box* :copy-worldcoords) :draw-on :flush t)
  (send *hrp2* :rarm :inverse-kinematics
        (send
         (send *box*
               :copy-worldcoords)
         :translate (float-vector 0 0 150)) ;adjust to fit the scissor size
        :rotation-axis nil
        :debug-view t)
  )


;;Function to put robot sim to temporary pick pose
(defun temp-pick-pose ()
  (setf ipose (float-vector 650 -320 900))
  (send *hrp2* :rarm :move-end-pos ipose :end-coords :worldpos)
  )


;; Function to clear previously drawn box
(defun remove-box ()
  (objects (remove *box* (send *irtviewer* :objects)))
  (setup-hrp2-tomato)
  )