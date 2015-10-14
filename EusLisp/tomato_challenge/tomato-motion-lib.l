;;
;; tomato-motion-lib.l
;;

(load "tomato-arrangement.l")
(load "tomato-basket-make.l")

(defun start-cut ()
  (hrp2w-start-grasp :rarm :angle -8))

(defun stop-cut ()
  (hrp2w-stop-grasp :rarm))

(defun move-hand-vi
  (&key (real *real*) (robot *hrp2*) (arm :rarm) (move-time (* 0.5 *base-time*))
	(move-dist 15) (move-headp t))
  (let (key-input velocity)
    (when move-headp
      (send *hrp2* :head :look-at (send *hrp2* arm :end-coords :worldpos)))
    (when real
      (send *ri* :angle-vector (send robot :angle-vector) (* 3 *base-time*)))
    (while t
      (setf key-input (read-line))
      (cond
       ((equal key-input "j") (setf velocity (float-vector 0 0 (- move-dist)))) ;; "j"=down
       ((equal key-input "k") (setf velocity (float-vector 0 0 move-dist))) ;; "k"=up
       ((equal key-input "h") (setf velocity (float-vector 0 move-dist 0))) ;; "h"=left
       ((equal key-input "l") (setf velocity (float-vector 0 (- move-dist) 0))) ;; "l"=right
       ((equal key-input "f") (setf velocity (float-vector move-dist 0 0))) ;; "f"=forward
       ((equal key-input "b") (setf velocity (float-vector (- move-dist) 0 0))) ;; "b"=backward
       ((equal key-input "q") (return-from move-hand-vi)) ;; "q"=quit
       )
      
      (print velocity)
      (print move-time)
      
      (send robot arm :move-end-pos velocity :world :rotation-axis :z)
      (when move-headp
	(send *hrp2* :head :look-at (send *hrp2* arm :end-coords :worldpos)))
      (send *irtviewer* :draw-objects)
      (when real
	(send *ri* :angle-vector (send robot :angle-vector) move-time))
      )
    ))

(unless (boundp '*hrp2*)
  (print ";; please load setup-hrp2-tomato.l"))
