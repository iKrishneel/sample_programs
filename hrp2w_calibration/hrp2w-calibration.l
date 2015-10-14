(load "calibration.l")

(defclass hrp2w-calibration-config
  :super calibration-config
  :slots (reset-angles))

(defmethod hrp2w-calibration-config
  (:init (&key ((:model-file amodel-file) nil)
               ((:model-class amodel-class) nil)
               ((:head-camera-config cam-conf) "/camera_remote/rgb")
               (camera-method nil))
    (send-super :init :model-file amodel-file :model-class amodel-class)
    (send self :head-camera-config cam-conf)
    (send self :add-board-object (make-mmurooka-board))
    (send self :add-chain-configs
          (list (cons "LARM" (butlast (send robot :larm)))
                (cons "RARM" (butlast (send robot :rarm)))
                (cons "LLEG" (send robot :lleg))
                ;;(cons "RLEG" (send robot :rleg))
                ;;(cons "head" (send robot :head))
                (cons "torso"
                      (send robot :torso))))
    ;; build good collision links
    (let ((collision-links (send robot :collision-check-pairs))
          (ignore-link-pairs nil))
          ;; (ignore-link-pairs '(("LARM_LINK6" . "LARM_LINK4")
          ;;                      ("RARM_LINK6" . "RARM_LINK4"))))
      (setq collision-links
            (remove-if #'(lambda (pair)
                           (let ((a (send (car pair) :name))
                                 (b (send (cdr pair) :name)))
                             (or (member (cons a b) ignore-link-pairs
                                         :test #'equal)
                                 (member (cons b a) ignore-link-pairs
                                         :test #'equal))))
                       collision-links))
      (send self :add-motion :larm
            (instance calibration-motion
                      :init
                      :prompt "Please put the checkerboard in the left hand"
                      :chain-ids '("LARM_chain" "head_chain")
                      :limb :larm
                      :collision-links collision-links
                      :robot-model robot
                      :reset-method :reset-pose
                      :camera-method camera-method
                      :filter-sampling-coords
                      '(lambda (c)
                         (> (elt (send c :worldpos) 1) 0))
                      :sampling-joint
                      (list
                       (instance sample-joint-information :init
                                 :joint (send robot :head :neck-p)
                                 :step-angle 15)
                       (instance sample-joint-information :init
                                 :joint (send robot :head :neck-y)
                                 :min-angle 0
                                 :step-angle 15))
                      :distances
                      (range-equivalents 520 640 60)
                      :angles
                      (range-equivalents (deg2rad -30) (deg2rad 120) (deg2rad 30))
                      :x-angles
                      (list (deg2rad -10) (deg2rad 0) (deg2rad 10))
                      :y-angles
                      (list (deg2rad -10) (deg2rad 0) (deg2rad 10))
                      :board (make-mmurooka-board)
                      :end-effector-offset
                      (let ((c (make-coords :pos (float-vector (+ 40) (+ 70 65 80) 0))))
                        (send c :rotate (deg2rad -90) :y)
                        (send c :rotate (deg2rad 90) :z)
                        c)))
      (send self :add-motion :rarm
            (instance calibration-motion
                      :init
                      :prompt "Please put the checkerboard in the right hand"
                      :chain-ids '("RARM_chain" "head_chain")
                      :limb :rarm
                      :angles           ;-90
                      ;;(list (deg2rad -30))
                      (range-equivalents (deg2rad -100) (deg2rad 0) (deg2rad 30))
                      :x-angles
                      (list (deg2rad -10) (deg2rad 0) (deg2rad 10))
                      :y-angles
                      (list (deg2rad -10) (deg2rad 0) (deg2rad 10))
                      :collision-links collision-links
                      :robot-model robot
                      :reset-method :reset-pose
                      :camera-method camera-method
                      :filter-sampling-coords
                      '(lambda (c)
                         (< (elt (send c :worldpos) 1) 0))
                      :sampling-joint
                      (list
                       (instance sample-joint-information :init
                                 :joint (send robot :head :neck-p)
                                 :step-angle 15)
                       (instance sample-joint-information :init
                                 :joint (send robot :head :neck-y)
                                 :step-angle 15))
                      :distances
                      (range-equivalents 520 720 60)
                      :board (make-mmurooka-board)
                      :end-effector-offset
                      (let ((c (make-coords
                                :pos (float-vector (+ 40) (- (+ 70 65 80)) 0))))
                        (send c :rotate (deg2rad -90) :y)
                        (send c :rotate (deg2rad 90) :z)
                        c)))
      self)))

 (defun generate-hrp2w-files ()
  (setq *calib* (instance hrp2w-calibration-config
                          :init
                          :model-file "package://hrpsys_ros_bridge_tutorials/euslisp/hrp2w-interface.l"
                          :model-class 'hrp2w-robot
                          :camera-method :CAMERA_C
			  ))
  (send *calib* :generate-motions)
  (send *calib* :generate-files "package://jsk_calibration/staro_calibration")
  )
