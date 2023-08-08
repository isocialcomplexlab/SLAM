; Auto-generated. Do not edit!


(cl:in-package custom_msgs-msg)


;//! \htmlinclude ObjectList.msg.html

(cl:defclass <ObjectList> (roslisp-msg-protocol:ros-message)
  ((objects
    :reader objects
    :initarg :objects
    :type (cl:vector custom_msgs-msg:WorldObject)
   :initform (cl:make-array 0 :element-type 'custom_msgs-msg:WorldObject :initial-element (cl:make-instance 'custom_msgs-msg:WorldObject)))
   (num
    :reader num
    :initarg :num
    :type cl:integer
    :initform 0))
)

(cl:defclass ObjectList (<ObjectList>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ObjectList>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ObjectList)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name custom_msgs-msg:<ObjectList> is deprecated: use custom_msgs-msg:ObjectList instead.")))

(cl:ensure-generic-function 'objects-val :lambda-list '(m))
(cl:defmethod objects-val ((m <ObjectList>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader custom_msgs-msg:objects-val is deprecated.  Use custom_msgs-msg:objects instead.")
  (objects m))

(cl:ensure-generic-function 'num-val :lambda-list '(m))
(cl:defmethod num-val ((m <ObjectList>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader custom_msgs-msg:num-val is deprecated.  Use custom_msgs-msg:num instead.")
  (num m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ObjectList>) ostream)
  "Serializes a message object of type '<ObjectList>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'objects))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'objects))
  (cl:let* ((signed (cl:slot-value msg 'num)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ObjectList>) istream)
  "Deserializes a message object of type '<ObjectList>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'objects) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'objects)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'custom_msgs-msg:WorldObject))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'num) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ObjectList>)))
  "Returns string type for a message object of type '<ObjectList>"
  "custom_msgs/ObjectList")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ObjectList)))
  "Returns string type for a message object of type 'ObjectList"
  "custom_msgs/ObjectList")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ObjectList>)))
  "Returns md5sum for a message object of type '<ObjectList>"
  "07c3c607e5f4dbf042b5d6e6584e7e64")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ObjectList)))
  "Returns md5sum for a message object of type 'ObjectList"
  "07c3c607e5f4dbf042b5d6e6584e7e64")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ObjectList>)))
  "Returns full string definition for message of type '<ObjectList>"
  (cl:format cl:nil "custom_msgs/WorldObject[] objects~%int32 num~%================================================================================~%MSG: custom_msgs/WorldObject~%string objClass~%float32 x~%float32 y~%float32 angle~%float64 prob~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ObjectList)))
  "Returns full string definition for message of type 'ObjectList"
  (cl:format cl:nil "custom_msgs/WorldObject[] objects~%int32 num~%================================================================================~%MSG: custom_msgs/WorldObject~%string objClass~%float32 x~%float32 y~%float32 angle~%float64 prob~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ObjectList>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'objects) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ObjectList>))
  "Converts a ROS message object to a list"
  (cl:list 'ObjectList
    (cl:cons ':objects (objects msg))
    (cl:cons ':num (num msg))
))
