for epoch in range(max_epochs):
    for batch_idx, minibatch_data in enumerate(data_loader()):
        #real data training
        discriminator.zero_grad()
        real_prediction = discriminator(minibatch_data)
        labels.data.fill_(1.0)
        real_loss = criterion(pred_real, labels)
        real_loss.backward()

        #fake data training
        fake_data = noise_sampler()
        fake_data = generator(fake_data).detach()
        fake_prediction = discriminator(fake_data)
        labels.data.fill_(0.0)
        fake_loss = criterion(fake_prediction, labels)
        fake_loss.backward()

        #update discriminator's weights
        discriminator_optimizer.step()     


        #train generator
        generator.zero_grad()
        gen_input = noise_sampler()
        generator_fake_data = generator(gen_input)
        discriminator_prediction = discriminator(generator_fake_data)
        labels.data.fill_(1.0)
        generator_loss = criterion(discriminator_prediction, labels) 
        generator_loss.backward()
        generator_optimizer.step()  

        # if print_every:
        #     print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
        #           % (epoch, max_epochs, batch_idx, len(train_loader),
        #              loss_d.data[0], loss_g.data[0]))

        #     if batch_idx % 100 == 0:
        #         vutils.save_image(data,
        #                 'samples_vanilla/real_samples.png')
        #         fake = generator(z)
        #         vutils.save_image(gen.data.view(batch_size, 1, 28, 28),
        #                 'samples_vanilla/fake_samples_epoch_%03d.png' % epoch)