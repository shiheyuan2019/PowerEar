from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='../results/iPhone14', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=30, help='how many test images to run')
        parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        parser.add_argument('--no_encode', action='store_true', help='do not produce encoded image')
        parser.add_argument('--sync',default=False, action='store_true', help='use the same latent code for different input images')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        
        self.isTrain = False
        return parser
