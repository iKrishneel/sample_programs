;;
;; setup-hrp2-tomato.l
;;

(load "package://hrpsys_ros_bridge_tutorials/euslisp/hrp2w-interface.l")

(defvar *real* nil)
(defvar *base-time* 1500)

(defun setup-hrp2-tomato ()
  (unless (boundp '*hrp2*)
    (hrp2w-init :sim t)
    (setq *hrp2* *hrp2w*)
    (setq ((send *hrp2w* :rarm :thumb-r) . max-angle) 15)
    )
  (send *hrp2w* :reset-manip-pose)
  (send *hrp2w* :head-neck-p :joint-angle 15)
  (send *hrp2w* :fix-leg-to-coords (make-coords))
  (objects *hrp2w*)
  )
(setup-hrp2-tomato)

(defun reset-tomato-pose ()
  (send *hrp2* :reset-manip-pose)
  (when (boundp '*ri*)
    (send *ri* :angle-vector (send *hrp2* :angle-vector) 5000))
  )

(load "tomato-motion-lib.l")
