;; Auto-generated. Do not edit!


(when (boundp 'custom_msgs::WorldObject)
  (if (not (find-package "CUSTOM_MSGS"))
    (make-package "CUSTOM_MSGS"))
  (shadow 'WorldObject (find-package "CUSTOM_MSGS")))
(unless (find-package "CUSTOM_MSGS::WORLDOBJECT")
  (make-package "CUSTOM_MSGS::WORLDOBJECT"))

(in-package "ROS")
;;//! \htmlinclude WorldObject.msg.html


(defclass custom_msgs::WorldObject
  :super ros::object
  :slots (_objClass _x _y _angle _prob ))

(defmethod custom_msgs::WorldObject
  (:init
   (&key
    ((:objClass __objClass) "")
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    ((:angle __angle) 0.0)
    ((:prob __prob) 0.0)
    )
   (send-super :init)
   (setq _objClass (string __objClass))
   (setq _x (float __x))
   (setq _y (float __y))
   (setq _angle (float __angle))
   (setq _prob (float __prob))
   self)
  (:objClass
   (&optional __objClass)
   (if __objClass (setq _objClass __objClass)) _objClass)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:angle
   (&optional __angle)
   (if __angle (setq _angle __angle)) _angle)
  (:prob
   (&optional __prob)
   (if __prob (setq _prob __prob)) _prob)
  (:serialization-length
   ()
   (+
    ;; string _objClass
    4 (length _objClass)
    ;; float32 _x
    4
    ;; float32 _y
    4
    ;; float32 _angle
    4
    ;; float64 _prob
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _objClass
       (write-long (length _objClass) s) (princ _objClass s)
     ;; float32 _x
       (sys::poke _x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _y
       (sys::poke _y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _angle
       (sys::poke _angle (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float64 _prob
       (sys::poke _prob (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _objClass
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _objClass (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _x
     (setq _x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _y
     (setq _y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _angle
     (setq _angle (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float64 _prob
     (setq _prob (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get custom_msgs::WorldObject :md5sum-) "02bf617586369c005f6429b5354bd9ab")
(setf (get custom_msgs::WorldObject :datatype-) "custom_msgs/WorldObject")
(setf (get custom_msgs::WorldObject :definition-)
      "string objClass
float32 x
float32 y
float32 angle
float64 prob
")



(provide :custom_msgs/WorldObject "02bf617586369c005f6429b5354bd9ab")


