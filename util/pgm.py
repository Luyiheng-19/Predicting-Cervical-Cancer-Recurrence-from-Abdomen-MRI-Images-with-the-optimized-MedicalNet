import paddle
import paddle.nn as nn
import copy


class TPGM(nn.Layer):
    def __init__(self, model, norm_mode, exclude_list=[]):
        super(TPGM, self).__init__()
        self.norm_mode = norm_mode
        self.exclude_list = exclude_list
        self.threshold = nn.Hardtanh(min=0, max=1)
        self.constraints_name = []
        self.constraints = nn.ParameterList([])
        self.create_contraint(model)  # Create constraint placeholders
        self.init = True

    def create_contraint(self, module):
        for name, para in module.named_parameters():
            if name not in self.exclude_list:
                self.constraints_name.append(name)
                # Create parameter
                temp = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0))
                # Set stop_gradient attribute separately, assuming you want this parameter to participate in gradient computation
                temp.stop_gradient = False  # Set to True if you don't want this parameter to participate in gradient computation
                self.constraints.append(temp)

    def apply_constraints(
            self,
            new,
            pre_trained,
            constraint_iterator,
            apply=False,
    ):
        for (name, new_para), anchor_para in zip(
                new.named_parameters(), pre_trained.parameters()
        ):
            if new_para.stop_gradient:
                continue
            if name not in self.exclude_list:
                alpha = self._project_ratio(
                    new_para,
                    anchor_para,
                    constraint_iterator,
                )
                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()
                if apply:
                    new_para.set_value(temp)
                else:
                    new_para.stop_gradient = True
                    new_para.set_value(temp)

        self.init = False

    def _project_ratio(self, new, anchor, constraint_iterator):
        t = new.detach() - anchor.detach()
        if "l2" in self.norm_mode:
            norms = paddle.norm(t)
        else:
            norms = paddle.sum(paddle.abs(t), axis=tuple(range(1, t.dim())), keepdim=True)

        constraint = next(constraint_iterator)

        if self.init:
            # Ensure temp is a tensor with shape [1]
            temp_value = norms.min() / 2 if norms.min() / 2 > 1e-8 else 1e-8
            temp = paddle.full(shape=[1], fill_value=temp_value)
            constraint.set_value(temp)
        else:
            # Ensure max is not less than min before calling paddle.clip
            clip_max = max(norms.max(), 1e-8)
            temp = paddle.clip(constraint, min=1e-8, max=clip_max)
            constraint.set_value(temp)

        ratio = self.threshold(constraint / (norms + 1e-8))
        return ratio

    def _clip(self, constraint, norms):
        return paddle.clip(constraint, 1e-8, norms.max())

    def forward(
            self,
            new=None,
            pre_trained=None,
            x=None,
            apply=False,
    ):
        constraint_iterator = iter(self.constraints)

        if apply:
            self.apply_constraints(
                new,
                pre_trained,
                constraint_iterator,
                apply=apply,
            )
        else:
            new_copy = copy.deepcopy(new)
            new_copy.eval()
            self.apply_constraints(new_copy, pre_trained, constraint_iterator)
            out = new_copy(x)
            return out


class tpgm_trainer(object):
    def __init__(
            self,
            model,
            pgmloader,
            norm_mode,
            proj_lr,
            max_iters,
            exclude_list=[]
    ) -> None:
        self.device = paddle.set_device('cpu')
        self.proj_lr = proj_lr
        self.max_iters = max_iters
        self.tpgm = TPGM(model, norm_mode=norm_mode, exclude_list=exclude_list).to(self.device)
        self.pre_trained = copy.deepcopy(model)
        self.pgm_optimizer = paddle.optimizer.Adam(parameters=self.tpgm.parameters(), learning_rate=self.proj_lr)
        self.pgmloader = pgmloader
        self.dataset_iterator = iter(self.pgmloader)
        self.criterion = nn.CrossEntropyLoss()

    def tpgm_iters(self, model, apply=False):
        if not apply:
            self.count = 0
            while self.count < self.max_iters:
                try:
                    data = next(self.dataset_iterator)
                except StopIteration:
                    self.dataset_iterator = iter(self.pgmloader)
                    data = next(self.dataset_iterator)

                pgm_image, pgm_target = data
                pgm_image = paddle.to_tensor(pgm_image).to(self.device)
                pgm_target = paddle.to_tensor(pgm_target).to(self.device)

                outputs = self.tpgm(model, self.pre_trained, x=pgm_image)
                pgm_loss = self.criterion(outputs, pgm_target)
                self.pgm_optimizer.clear_grad()
                pgm_loss.backward()
                self.pgm_optimizer.step()
                self.count += 1

                if (self.count + 1) % 20 == 0:
                    print("{}/{} completed".format(self.count, self.max_iters))

        self.tpgm(model, self.pre_trained, apply=True)
