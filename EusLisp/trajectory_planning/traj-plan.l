
;; set up fixed size variables
(defvar *tomato-size* 50)
(defvar *hako-y-lenght* 350)
(defvar *hako-x-width* 250)


;; global variable set default to 30
;(defvar *y-lenght* 30)
(defvar default-start 30)
(defvar *padding* 10)

(setq *y-lenght* default-start)
(setq *x-lenght* default-start)

(defun place-tomato ()
  (if (< *y-lenght* *hako-y-lenght* )
      (setf *y-lenght* (+ (+ *tomato-size* *padding*) *y-lenght*)))
  
    ;;(when (>= *y-lenght* *hako-y-lenght* )
     ;;(if (>= (+ *x-lenght* *padding*) *hako-x-width*)
					;	 (setq *x-lenght* 30))
     
     ;(setf *y-lenght* default-start)
      ;(setf *x-lenght* (+ (+ *tomato-size* *padding* ) *x-lenght*)))
     )
