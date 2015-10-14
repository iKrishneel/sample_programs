(load "package://hrpsys_ros_bridge/euslisp/rtm-ros-robot-interface.l")
(require :hrp2w "package://hrpsys_ros_bridge_tutorials/models/hrp2w.l")
(when (probe-file (ros::resolve-ros-path "package://hrpsys_ros_bridge_tutorials/euslisp/hrp2w-utils.l"))
  (require :hrp2w-utils "package://hrpsys_ros_bridge_tutorials/euslisp/hrp2w-utils.l"))
(load "package://hrpsys_ros_bridge_tutorials/euslisp/hrp2-common-interface.l")
(if (ros::resolve-ros-path "package://jsk_hrp2_ros_bridge")
    (ros::load-ros-manifest "jsk_hrp2_ros_bridge"))

(defclass hrp2w-interface
  :super hrp2-common-interface
  :slots ())
(defmethod hrp2w-interface
  (:init (&rest args)
         (send-super* :init :robot hrp2w-robot args)))

(defun hrp2jsknt-init (&rest args)
  (if (not (boundp '*ri*))
      (setq *ri* (instance* hrp2w-interface :init args)))
  (if (not (boundp '*hrp2w*))
      (setq *hrp2w* (instance hrp2w-robot :init))))
