
(setq *counter* 0)
(setq *pos_counter* 0)
(setq board-object "Chessboard10x7")
(setq controller-type "rarm_controller")
(setq chain-ids '("RARM_chain" "head_chain"))
(setq tm 10)
(setq DEFAULT-ARM-NAME "RARM")

(defun generate-yaml(arm-name angle-vector )
    (with-open-file
     (f (format nil "samples/~A_~0,4d.yaml" (string arm-name) *counter*)
	:direction :output
	:if-exists :supersede
	:if-does-not-exist :create)

     (format f "camera_measurements:~%")
     (format f "- {cam_id: head_camera, config: ~A}~%"
	     (string-downcase (string "Chessboard10x7")))
     (format f "joint_commands:~%")
     (format f "- controller: ~A~%" (string-downcase (string controller-type)))
     (format f "  segments:~%")
     (format f "  - duration: ~A~%" tm)
     (format f "    positions: ~A~%" (print-array ;convert to radian
                                        (mapcar #'deg2rad (coerce angle-vector cons))))
     (format f "joint_measurements:~%")
     (dolist (chain-id chain-ids)
       (format f "- {chain_id: ~A,  config: tight_tol}~%" chain-id))

     (format f "sample_id: ~A_~0,4d~%" (string arm-name) *counter*)
     (format f "target: {chain_id: ~A_chain, target_id: ~A}~%"
	     (string arm-name) (string-downcase (string  board-object )))

     (setq *counter* (+ *counter* 1)))
  )

(defun print-array (angle-vector)
  (let ((ss (make-string-output-stream)))
    (format ss "[")
    (dotimes (i (length angle-vector))
      (format ss "~A" (elt angle-vector i))
      (unless (= i (1- (length angle-vector)))
        (format ss ", "))
      )
    (format ss "]")
    (get-output-stream-string ss)))



(defun move-hrp2w-simu (&optional arm-name)
  ;;(unless (boundp arm-name)
  ;;  (setq DEFAULT-ARM-NAME (string (arm-name)) ))
  (move-hand-vi)
  (setq angle-vector (send *hrp2w* :angle-vector))
  (generate-yaml DEFAULT-ARM-NAME angle-vector))


(defun move-hrp2w-real ()
  ;(move-hand-vi)
  (setq angle-vector (send *ri* :state :angle-vector))
  (generate-yaml DEFAULT-ARM-NAME angle-vector)
  
  (setq position-vector (send *hrp2w* :rarm :end-coords :copy-worldcoords))
  (save-position DEFAULT-ARM-NAME position-vector))




(defun save-position(arm-name p-vector )
    (with-open-file
     (f (format nil "samples/position/~A_~0,4d.yaml" (string arm-name) *pos_counter*)
	:direction :output
	:if-exists :supersede
	:if-does-not-exist :create)

     (format f "~A"  p-vector)
     (setq *pos_counter* (+ *pos_counter* 1))))
