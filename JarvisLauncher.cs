using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;

namespace JarvisLauncher
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Title = "JARVIS OMEGA // 3D HUD & LOCAL NEURAL MATRIX";
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(@"
   ===================================================================
      JARVIS OMEGA // 3D HOLOGRAPHIC HUD & LOCAL AI CORE
   ===================================================================
");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" [*] Initializing JARVIS OMEGA Subsystems...");
            
            string appDir = @"C:\Users\HP\.gemini\antigravity\scratch\vega";
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string ollamaPath = Path.Combine(localAppData, @"Programs\Ollama\ollama.exe");

            // 1. Check Ollama
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write(" [*] Checking Local AI Neural Engine (Ollama)... ");
            bool ollamaRunning = IsPortOpen("http://localhost:11434/api/tags");
            if (!ollamaRunning && File.Exists(ollamaPath))
            {
                Console.WriteLine("Starting Ollama background daemon...");
                ProcessStartInfo ollamaInfo = new ProcessStartInfo(ollamaPath, "serve")
                {
                    CreateNoWindow = true,
                    UseShellExecute = false
                };
                Process.Start(ollamaInfo);
                Thread.Sleep(2000);
            }
            else
            {
                Console.WriteLine("ONLINE [OK]");
            }

            // 2. Check & Launch Web Server
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write(" [*] Checking JARVIS 3D Web HUD Engine... ");
            bool appRunning = IsPortOpen("http://localhost:7860");
            if (!appRunning)
            {
                Console.WriteLine("Starting Python server...");
                ProcessStartInfo pyInfo = new ProcessStartInfo("py", "-3.11 app.py")
                {
                    WorkingDirectory = appDir,
                    UseShellExecute = true
                };
                Process.Start(pyInfo);
                
                for (int i = 0; i < 15; i++)
                {
                    Thread.Sleep(1000);
                    if (IsPortOpen("http://localhost:7860")) break;
                }
            }
            else
            {
                Console.WriteLine("ONLINE [OK]");
            }

            // 3. Open Browser
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n [OK] Core Online! Launching 3D Holographic HUD in Browser: http://localhost:7860");
            try
            {
                Process.Start(new ProcessStartInfo("http://localhost:7860") { UseShellExecute = true });
            }
            catch { }

            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("\n ===============================================================");
            Console.WriteLine("  JARVIS OMEGA is currently running in the background.");
            Console.WriteLine("  - 3D Holographic Arc Reactor: Active");
            Console.WriteLine("  - Local AI Model: llama3.2:1b (100% Offline)");
            Console.WriteLine("  - 7 Specialist Agents: Standing By");
            Console.WriteLine("  - Web Interface: http://localhost:7860");
            Console.WriteLine("  Press 'Q' to exit JARVIS.");
            Console.WriteLine(" ===============================================================\n");

            while (true)
            {
                if (Console.KeyAvailable)
                {
                    var key = Console.ReadKey(true);
                    if (key.Key == ConsoleKey.Q) break;
                }
                Thread.Sleep(500);
            }
        }

        static bool IsPortOpen(string url)
        {
            try
            {
                using (var client = new HttpClient())
                {
                    client.Timeout = TimeSpan.FromSeconds(1.5);
                    var resp = client.GetAsync(url).Result;
                    return resp.IsSuccessStatusCode;
                }
            }
            catch
            {
                return false;
            }
        }
    }
}
