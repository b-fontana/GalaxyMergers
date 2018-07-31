def add_args(parser):
   parser.add_argument(
      '--mode',
      type=str,
      default='train',
      help='Mode. Currently available options: \n 1) save \n 2) train \n 3) predict'
   )
   parser.add_argument(
      '--save_model_name',
      type=str,
      help="Name of the file where the model is going to be saved. It must have a 'h5' extension."
   )
   parser.add_argument(
      '--data_to_convert',
      nargs='+',
      type=str,
      help="Name of the folder where the pictures to be converted to the '.tfrecord' format are stored. It is assumed that that folder can be found in /data1/alves/."
   )
   parser.add_argument(
      '--saved_model_name',
      type=str,
      help="Name of the file where the model was saved. It must have a 'h5' extension."
   )
   parser.add_argument(
      '--save_data_name',
      type=str,
      help="File name where the data is going to be saved. It must have a 'tfrecord' extension."
   )
   parser.add_argument(
      '--saved_train_data',
      nargs='+',
      type=str,
      help="File name(s) where the training data was saved. They must have a 'tfrecord' extension."
   )
   parser.add_argument(
      '--saved_valid_data',
      nargs='+',
      type=str,
      help="File name(s) where the validation data was saved. They must have a 'tfrecord' extension."
   )
   parser.add_argument(
      '--cutmin',
      type=int,
      default=0,
      help='Left range of the array of pictures to be saved for each class.'
   )
   parser.add_argument(
      '--cutmax',
      type=int,
      default=999999,
      help='Right range of the array of pictures to be saved for each class.'
   )
   parser.add_argument(
      '--tensorboard',
      type=str,
      help='Right range of the array of pictures to be saved for each class.'
   )
   parser.add_argument(
      '--input_depth',
      type=int,
      default=3,
      help="Select '1' for grey-scale or '3' for RGB. This does not perform any convertion; it just slices the numpy array such that only the first column is considered when evaluating the tensors."
   )
   parser.add_argument(
      '--file_metrics',
      type=str,
      help='Name of the file where the metrics are stored.'
   )
   parser.add_argument(
      '--prediction_file',
      type=str,
      help="File name for prediction."
   )
   parser.add_argument(
      '--prediction_sr',
      nargs='+',
      type=str,
      help="Size ratio for checking the predictions."
   )
   parser.add_argument(
      '--prediction_mr',
      nargs='+',
      type=str,
      help="Mass ratio for checking the predictions."
  )
   return parser.parse_known_args()
