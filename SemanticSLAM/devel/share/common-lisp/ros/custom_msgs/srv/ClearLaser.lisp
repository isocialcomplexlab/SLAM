; Auto-generated. Do not edit!


(cl:in-package custom_msgs-srv)


;//! \htmlinclude ClearLaser-request.msg.html

(cl:defclass <ClearLaser-request> (roslisp-msg-protocol:ros-message)
  ((weight
    :reader weight
    :initarg :weight
    :type cl:fixnum
    :initform 0)
   (laser
    :reader laser
    :initarg :laser
    :type sensor_msgs-msg:LaserScan
    :initform (cl:make-instance 'sensor_msgs-msg:LaserScan)))
)

(cl:defclass ClearLaser-request (<ClearLaser-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ClearLaser-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ClearLaser-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name custom_msgs-srv:<ClearLaser-request> is deprecated: use custom_msgs-srv:ClearLaser-request instead.")))

(cl:ensure-generic-function 'weight-val :lambda-list '(m))
(cl:defmethod weight-val ((m <ClearLaser-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader custom_msgs-srv:weight-val is deprecated.  Use custom_msgs-srv:weight instead.")
  (weight m))

(cl:ensure-generic-function 'laser-val :lambda-list '(m))
(cl:defmethod laser-val ((m <ClearLaser-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader custom_msgs-srv:laser-val is deprecated.  Use custom_msgs-srv:laser instead.")
  (laser m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ClearLaser-request>) ostream)
  "Serializes a message object of type '<ClearLaser-request>"
  (cl:let* ((signed (cl:slot-value msg 'weight)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'laser) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ClearLaser-request>) istream)
  "Deserializes a message object of type '<ClearLaser-request>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'weight) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'laser) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ClearLaser-request>)))
  "Returns string type for a service object of type '<ClearLaser-request>"
  "custom_msgs/ClearLaserRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ClearLaser-request)))
  "Returns string type for a service object of type 'ClearLaser-request"
  "custom_msgs/ClearLaserRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ClearLaser-request>)))
  "Returns md5sum for a message object of type '<ClearLaser-request>"
  "f427fbbf8b1198bb81f85f08fbb4a779")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ClearLaser-request)))
  "Returns md5sum for a message object of type 'ClearLaser-request"
  "f427fbbf8b1198bb81f85f08fbb4a779")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ClearLaser-request>)))
  "Returns full string definition for message of type '<ClearLaser-request>"
  (cl:format cl:nil "# svr for the ClearCells service~%int8 weight~%sensor_msgs/LaserScan laser~%~%================================================================================~%MSG: sensor_msgs/LaserScan~%# Single scan from a planar laser range-finder~%#~%# If you have another ranging device with different behavior (e.g. a sonar~%# array), please find or create a different message, since applications~%# will make fairly laser-specific assumptions about this data~%~%Header header            # timestamp in the header is the acquisition time of ~%                         # the first ray in the scan.~%                         #~%                         # in frame frame_id, angles are measured around ~%                         # the positive Z axis (counterclockwise, if Z is up)~%                         # with zero angle being forward along the x axis~%                         ~%float32 angle_min        # start angle of the scan [rad]~%float32 angle_max        # end angle of the scan [rad]~%float32 angle_increment  # angular distance between measurements [rad]~%~%float32 time_increment   # time between measurements [seconds] - if your scanner~%                         # is moving, this will be used in interpolating position~%                         # of 3d points~%float32 scan_time        # time between scans [seconds]~%~%float32 range_min        # minimum range value [m]~%float32 range_max        # maximum range value [m]~%~%float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)~%float32[] intensities    # intensity data [device-specific units].  If your~%                         # device does not provide intensities, please leave~%                         # the array empty.~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ClearLaser-request)))
  "Returns full string definition for message of type 'ClearLaser-request"
  (cl:format cl:nil "# svr for the ClearCells service~%int8 weight~%sensor_msgs/LaserScan laser~%~%================================================================================~%MSG: sensor_msgs/LaserScan~%# Single scan from a planar laser range-finder~%#~%# If you have another ranging device with different behavior (e.g. a sonar~%# array), please find or create a different message, since applications~%# will make fairly laser-specific assumptions about this data~%~%Header header            # timestamp in the header is the acquisition time of ~%                         # the first ray in the scan.~%                         #~%                         # in frame frame_id, angles are measured around ~%                         # the positive Z axis (counterclockwise, if Z is up)~%                         # with zero angle being forward along the x axis~%                         ~%float32 angle_min        # start angle of the scan [rad]~%float32 angle_max        # end angle of the scan [rad]~%float32 angle_increment  # angular distance between measurements [rad]~%~%float32 time_increment   # time between measurements [seconds] - if your scanner~%                         # is moving, this will be used in interpolating position~%                         # of 3d points~%float32 scan_time        # time between scans [seconds]~%~%float32 range_min        # minimum range value [m]~%float32 range_max        # maximum range value [m]~%~%float32[] ranges         # range data [m] (Note: values < range_min or > range_max should be discarded)~%float32[] intensities    # intensity data [device-specific units].  If your~%                         # device does not provide intensities, please leave~%                         # the array empty.~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ClearLaser-request>))
  (cl:+ 0
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'laser))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ClearLaser-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ClearLaser-request
    (cl:cons ':weight (weight msg))
    (cl:cons ':laser (laser msg))
))
;//! \htmlinclude ClearLaser-response.msg.html

(cl:defclass <ClearLaser-response> (roslisp-msg-protocol:ros-message)
  ((b
    :reader b
    :initarg :b
    :type cl:float
    :initform 0.0))
)

(cl:defclass ClearLaser-response (<ClearLaser-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ClearLaser-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ClearLaser-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name custom_msgs-srv:<ClearLaser-response> is deprecated: use custom_msgs-srv:ClearLaser-response instead.")))

(cl:ensure-generic-function 'b-val :lambda-list '(m))
(cl:defmethod b-val ((m <ClearLaser-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader custom_msgs-srv:b-val is deprecated.  Use custom_msgs-srv:b instead.")
  (b m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ClearLaser-response>) ostream)
  "Serializes a message object of type '<ClearLaser-response>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'b))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ClearLaser-response>) istream)
  "Deserializes a message object of type '<ClearLaser-response>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'b) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ClearLaser-response>)))
  "Returns string type for a service object of type '<ClearLaser-response>"
  "custom_msgs/ClearLaserResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ClearLaser-response)))
  "Returns string type for a service object of type 'ClearLaser-response"
  "custom_msgs/ClearLaserResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ClearLaser-response>)))
  "Returns md5sum for a message object of type '<ClearLaser-response>"
  "f427fbbf8b1198bb81f85f08fbb4a779")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ClearLaser-response)))
  "Returns md5sum for a message object of type 'ClearLaser-response"
  "f427fbbf8b1198bb81f85f08fbb4a779")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ClearLaser-response>)))
  "Returns full string definition for message of type '<ClearLaser-response>"
  (cl:format cl:nil "float32 b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ClearLaser-response)))
  "Returns full string definition for message of type 'ClearLaser-response"
  (cl:format cl:nil "float32 b~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ClearLaser-response>))
  (cl:+ 0
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ClearLaser-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ClearLaser-response
    (cl:cons ':b (b msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ClearLaser)))
  'ClearLaser-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ClearLaser)))
  'ClearLaser-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ClearLaser)))
  "Returns string type for a service object of type '<ClearLaser>"
  "custom_msgs/ClearLaser")