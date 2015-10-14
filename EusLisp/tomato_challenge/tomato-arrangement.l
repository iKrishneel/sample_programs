;;
;; Program for placement of Tomato
;;

;**  To get the current joint state of the robot pose use function
;**  (send *hrp2* :rarm (joint name) : end-coords); 
;        to get the vector only add ":worldpos" 

;; Size of tomato + tolerance 10
(defvar tomato-diameter 60)
(defvar *tomato-box-size-y* 330)
(defvar *tomato-box-size-x* 230)

;; Function to plan the main trajectory planning
(defun start-placement ()
  (setf rarm-vector (send *hrp2* :rarm :end-coords :worldpos))
  (print rarm-vector)
  
  )


; Function to keep track of the position of the last placed tomato
(defun tomato-basket-arrangement ()
  
  )

; Steps to Placement
; 1) Estimate the current right arm end coordinate
; 2) Estimate the position to place the tomato
