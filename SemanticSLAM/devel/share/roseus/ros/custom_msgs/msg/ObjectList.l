;; Auto-generated. Do not edit!


(when (boundp 'custom_msgs::ObjectList)
  (if (not (find-package "CUSTOM_MSGS"))
    (make-package "CUSTOM_MSGS"))
  (shadow 'ObjectList (find-package "CUSTOM_MSGS")))
(unless (find-package "CUSTOM_MSGS::OBJECTLIST")
  (make-package "CUSTOM_MSGS::OBJECTLIST"))

(in-package "ROS")
;;//! \htmlinclude ObjectList.msg.html


(defclass custom_msgs::ObjectList
  :super ros::object
  :slots (_objects _num ))

(defmethod custom_msgs::ObjectList
  (:init
   (&key
    ((:objects __objects) (let (r) (dotimes (i 0) (push (instance custom_msgs::WorldObject :init) r)) r))
    ((:num __num) 0)
    )
   (send-super :init)
   (setq _objects __objects)
   (setq _num (round __num))
   self)
  (:objects
   (&rest __objects)
   (if (keywordp (car __objects))
       (send* _objects __objects)
     (progn
       (if __objects (setq _objects (car __objects)))
       _objects)))
  (:num
   (&optional __num)
   (if __num (setq _num __num)) _num)
  (:serialization-length
   ()
   (+
    ;; custom_msgs/WorldObject[] _objects
    (apply #'+ (send-all _objects :serialization-length)) 4
    ;; int32 _num
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; custom_msgs/WorldObject[] _objects
     (write-long (length _objects) s)
     (dolist (elem _objects)
       (send elem :serialize s)
       )
     ;; int32 _num
       (write-long _num s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; custom_msgs/WorldObject[] _objects
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _objects (let (r) (dotimes (i n) (push (instance custom_msgs::WorldObject :init) r)) r))
     (dolist (elem- _objects)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; int32 _num
     (setq _num (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get custom_msgs::ObjectList :md5sum-) "07c3c607e5f4dbf042b5d6e6584e7e64")
(setf (get custom_msgs::ObjectList :datatype-) "custom_msgs/ObjectList")
(setf (get custom_msgs::ObjectList :definition-)
      "custom_msgs/WorldObject[] objects
int32 num
================================================================================
MSG: custom_msgs/WorldObject
string objClass
float32 x
float32 y
float32 angle
float64 prob
")



(provide :custom_msgs/ObjectList "07c3c607e5f4dbf042b5d6e6584e7e64")


