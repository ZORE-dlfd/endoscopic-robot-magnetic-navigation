using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Forms;

internal static class Launcher
{
    [STAThread]
    private static int Main(string[] args)
    {
        bool extractOnly = args.Length > 0 && args[0] == "--extract-only";
        try
        {
            string directory = extractOnly && args.Length > 1
                ? Path.GetFullPath(args[1])
                : Path.Combine(Path.GetTempPath(), "MagneticCapsuleCamera", PayloadInfo.Sha256.Substring(0, 16));
            Directory.CreateDirectory(directory);
            string target = Path.Combine(directory, "Magnetic_Camera.html");
            if (!File.Exists(target) || HashFile(target) != PayloadInfo.Sha256)
            {
                string temporary = Path.Combine(directory, Guid.NewGuid().ToString("N") + ".tmp");
                try
                {
                    using (Stream source = Assembly.GetExecutingAssembly().GetManifestResourceStream("Magnetic.Html.gz"))
                    using (GZipStream gzip = new GZipStream(source, CompressionMode.Decompress))
                    using (FileStream output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                        gzip.CopyTo(output);
                    if (HashFile(temporary) != PayloadInfo.Sha256)
                        throw new InvalidDataException("内嵌页面校验失败，请重新获取文件。");
                    if (File.Exists(target)) File.Delete(target);
                    File.Move(temporary, target);
                }
                finally
                {
                    if (File.Exists(temporary)) File.Delete(temporary);
                }
            }
            File.WriteAllText(Path.Combine(directory, "extraction.sha256"), HashFile(target), Encoding.UTF8);
            if (extractOnly) return 0;
            string url = new Uri(target).AbsoluteUri;
            string[] edgeCandidates = new string[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft", "Edge", "Application", "msedge.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Microsoft", "Edge", "Application", "msedge.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Microsoft", "Edge", "Application", "msedge.exe")
            };
            foreach (string edge in edgeCandidates)
            {
                if (!File.Exists(edge)) continue;
                try
                {
                    Process.Start(new ProcessStartInfo(edge, "--app=\"" + url + "\" --window-size=1680,1080") { UseShellExecute = false });
                    return 0;
                }
                catch (System.ComponentModel.Win32Exception) { }
            }
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            return 0;
        }
        catch (Exception error)
        {
            if (!extractOnly)
                MessageBox.Show("无法打开交互页面：\n" + error.Message + "\n也可以直接打开旁边的 HTML 文件。", "Magnetic Viewer", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    private static string HashFile(string filename)
    {
        using (SHA256 algorithm = SHA256.Create())
        using (FileStream input = File.OpenRead(filename))
            return BitConverter.ToString(algorithm.ComputeHash(input)).Replace("-", "").ToLowerInvariant();
    }
}
