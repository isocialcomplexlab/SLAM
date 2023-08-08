;; Auto-generated. Do not edit!


(when (boundp 'custom_msgs::ClearLaser)
  (if (not (find-package "CUSTOM_MSGS"))
    (make-package "CUSTOM_MSGS"))
  (shadow 'ClearLaser (find-package "CUSTOM_MSGS")))
(unless (find-package "CUSTOM_MSGS::CLEARLASER")
  (make-package "CUSTOM_MSGS::CLEARLASER"))
(unless (find-package "CUSTOM_MSGS::CLEARLASERREQUEST")
  (make-package "CUSTOM_MSGS::CLEARLASERREQUEST"))
(unless (find-package "CUSTOM_MSGS::CLEARLASERRESPONSE")
  (make-package "CUSTOM_MSGS::CLEARLASERRESPONSE"))

(in-package "ROS")

(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))




(defclass custom_msgs::ClearLaserRequest
  :super ros::object
  :slots (_weight _laser ))

(defmethod custom_msgs::ClearLaserRequest
  (:init
   (&key
    ((:weight __weight) 0)
    ((:laser __laser) (instance sensor_msgs::LaserScan :init))
    )
   (send-super :init)
   (setq _weight (round __weight))
   (setq _laser __laser)
   self)
  (:weight
   (&optional __weight)
   (if __weight (setq _weight __weight)) _weight)
  (:laser
   (&rest __laser)
   (if (keywordp (car __laser))
       (send* _laser __laser)
     (progn
       (if __laser (setq _laser (car __laser)))
       _laser)))
  (:serialization-length
   ()
   (+
    ;; int8 _weight
    1
    ;; sensor_msgs/LaserScan _laser
    (send _laser :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int8 _weight
       (write-byte _weight s)
     ;; sensor_msgs/LaserScan _laser
       (send _laser :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int8 _weight
     (setq _weight (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _weight 127) (setq _weight (- _weight 256)))
   ;; sensor_msgs/LaserScan _laser
     (send _laser :deserialize buf ptr-) (incf ptr- (send _laser :serialization-length))
   ;;
   self)
  )

(defclass custom_msgs::ClearLaserResponse
  :super ros::object
  :slots (_b ))

(defmethod custom_msgs::ClearLaserResponse
  (:init
   (&key
    ((:b __b) 0.0)
    )
   (send-super :init)
   (setq _b (float __b))
   self)
  (:b
   (&optional __b)
   (if __b (setq _b __b)) _b)
  (:serialization-length
   ()
   (+
    ;; float32 _b
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _b
       (sys::poke _b (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _b
     (setq _b (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass custom_msgs::ClearLaser
  :super ros::object
  :slots ())

(setf (get custom_msgs::ClearLaser :md5sum-) "f427fbbf8b1198bb81f85f08fbb4a779")
(setf (get custom_msgs::ClearLaser :datatype-) "custom_msgs/ClearLaser")
(setf (get custom_msgs::ClearLaser :request) custom_msgs::ClearLaserRequest)
(setf (get custom_msgs::ClearLaser :response) custom_msgs::ClearLaserResponse)

(defmethod custom_msgs::ClearLaserRequest
  (:response () (instance custom_msgs::ClearLaserResponse :init)))

(setf (get custom_msgs::ClearLaserRequest :md5sum-) "f427fbbf8b1198bb81f85f08fbb4a779")
(setf (get custom_msgs::ClearLaserRequest :datatype-) "custom_msgs/ClearLaserRequest")
(setf (get custom_msgs::ClearLaserRequest :definition-)
      "# svr for the ClearCells service
int8 weight
sensor_msgs/LaserScan laser

================================================================================
MSG: sensor_msgs/LaserScan
# Single scan from a planar laser range-finder
#
# If you have another ranging device with different behavior (e.g. a sonar
# array), please find or create a different message, since applications
# will make fairly laser-specific assumptions about this data

Header header            # timestamp in the header is the acquisition time of 
                         # the first ray in the scan.
                         #
                         # in frame frame_id, angles are measured around 
                         # the positive Z axis (counterclockwise, if Z is up)
                         # with zero angle being forward along the x axis
                         
float32 angle_min        # start angle of the scan [rad]
float32 angle_max        # end angle of the scan [rad]
float32 angle_increment  # angular distance between measurements [rad]

float32 time_increment   # time between measurements [seconds] - if your scanner
                         # is moving, this will be used in interpolating position
                         # of 3d points
float32 scan_time        # time between scans [seconds]

float32 range_min        # minimum range value [m]
float32 range_max        # maximum range value [m]

float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)
float32[] intensities    # intensity data [device-specific units].  If your
                         # device does not provide intensities, please leave
                         # the array empty.

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id
---
float32 b
")

(setf (get custom_msgs::ClearLaserResponse :md5sum-) "f427fbbf8b1198bb81f85f08fbb4a779")
(setf (get custom_msgs::ClearLaserResponse :datatype-) "custom_msgs/ClearLaserResponse")
(setf (get custom_msgs::ClearLaserResponse :definition-)
      "# svr for the ClearCells service
int8 weight
sensor_msgs/LaserScan laser

================================================================================
MSG: sensor_msgs/LaserScan
# Single scan from a planar laser range-finder
#
# If you have another ranging device with different behavior (e.g. a sonar
# array), please find or create a different message, since applications
# will make fairly laser-specific assumptions about this data

Header header            # timestamp in the header is the acquisition time of 
                         # the first ray in the scan.
                         #
                         # in frame frame_id, angles are measured around 
                         # the positive Z axis (counterclockwise, if Z is up)
                         # with zero angle being forward along the x axis
                         
float32 angle_min        # start angle of the scan [rad]
float32 angle_max        # end angle of the scan [rad]
float32 angle_increment  # angular distance between measurements [rad]

float32 time_increment   # time between measurements [seconds] - if your scanner
                         # is moving, this will be used in interpolating position
                         # of 3d points
float32 scan_time        # time between scans [seconds]

float32 range_min        # minimum range value [m]
float32 range_max        # maximum range value [m]

float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)
float32[] intensities    # intensity data [device-specific units].  If your
                         # device does not provide intensities, please leave
                         # the array empty.

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id
---
float32 b
")



(provide :custom_msgs/ClearLaser "f427fbbf8b1198bb81f85f08fbb4a779")


