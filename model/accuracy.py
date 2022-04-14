from utils.useful_functions import print_stats


#We perform tensorboard visualisation by writing the stats to the writer
def write_to_writer(writer, args, i_epoch, list_with_losses, train):
    TESTCOLOR = '\033[104m'
    TRAINCOLOR = '\033[100m'
    NORMALCOLOR = '\033[0m'

    if train:
        loss_train, loss_train_3d, loss_train_raster, loss_train_logl, loss_train_ent = list_with_losses
        if args.logl:
            print(
                TRAINCOLOR + 'Epoch %3d -> Train Loss: %1.4f Train Loss 3D: %1.4f Train Loss Raster: %1.4f Train Loss Log: %1.4f Train Loss Ent: %1.4f' % (
                i_epoch + 1, loss_train, loss_train_3d, loss_train_raster, loss_train_logl, loss_train_ent) + NORMALCOLOR)
            print_stats(args.stats_file, 'Epoch %3d -> Train Loss: %1.4f Train Loss 3D  : %1.4f Train Loss Raster: %1.4f Train Loss Log: %1.4f Train Loss Ent: %1.4f' % (
                i_epoch + 1, loss_train, loss_train_3d, loss_train_raster, loss_train_logl, loss_train_ent), print_to_console=False)
            writer.add_scalar('Loss/train_logl', loss_train_logl, i_epoch + 1)
        else:
            print(TRAINCOLOR + 'Epoch %3d -> Train Loss: %1.4f Train Loss 3D: %1.4f Train Loss Ent: %1.4f' % (
            i_epoch + 1, loss_train, loss_train_3d, loss_train_ent) + NORMALCOLOR)
            print_stats(args.stats_file,
                        'Epoch %3d -> Train Loss: %1.4f Train Loss 3D: %1.4f Train Loss Ent: %1.4f' % (
                            i_epoch + 1, loss_train, loss_train_3d, loss_train_ent), print_to_console=False)
        writer.add_scalar('Loss/train', loss_train, i_epoch + 1)
        writer.add_scalar('Loss/train_3D', loss_train_3d, i_epoch + 1)
        writer.add_scalar('Loss/train_raster', loss_train_raster, i_epoch + 1)
        writer.add_scalar('Loss/train_ent', loss_train_ent, i_epoch + 1)

    else:
        loss_test, loss_test_3d, loss_test_raster, loss_test_logl, loss_test_ent = list_with_losses
        if args.logl:
            print(
                TESTCOLOR + 'Test Loss: %1.4f Test Loss 3D: %1.4f Test Loss Raster: %1.4f Test Loss Log: %1.4f Test Loss Ent: %1.4f' % (
                loss_test, loss_test_3d, loss_test_raster, loss_test_logl, loss_test_ent) + NORMALCOLOR)
            print_stats(args.stats_file,
                        'Test Loss: %1.4f Test Loss 3D: %1.4f Test Loss Raster: %1.4f Test Loss Log: %1.4f Test Loss Ent: %1.4f' % (
                loss_test, loss_test_3d, loss_test_raster, loss_test_logl, loss_test_ent), print_to_console=False)
            writer.add_scalar('Loss/train_logl', loss_test_logl, i_epoch + 1)
        else:
            print(TESTCOLOR + 'Test Loss: %1.4f Test Loss 3D: %1.4f Test Loss Raster: %1.4f Test Loss Log: %1.4f' % (
                loss_test, loss_test_3d, loss_test_raster, loss_test_ent) + NORMALCOLOR)
            print_stats(args.stats_file,
                        'Test Loss: %1.4f Test Loss 3D: %1.4f Test Loss Raster: %1.4f Test Loss Log: %1.4f' % (
                loss_test, loss_test_3d, loss_test_raster, loss_test_ent), print_to_console=False)
        writer.add_scalar('Loss/test', loss_test, i_epoch + 1)
        writer.add_scalar('Loss/test_3D', loss_test_3d, i_epoch + 1)
        writer.add_scalar('Loss/test_raster', loss_test_raster, i_epoch + 1)
        writer.add_scalar('Loss/test_log', loss_test_ent, i_epoch + 1)
    return writer